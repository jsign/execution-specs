"""
State.

.. contents:: Table of Contents
    :backlinks: none
    :local:

Introduction
------------

The state contains all information that is preserved between transactions.

It consists of a main account trie and storage tries for each contract.

There is a distinction between an account that does not exist and
`EMPTY_ACCOUNT`.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

from ethereum_types.bytes import Bytes, Bytes32
from ethereum_types.frozen import modify
from ethereum_types.numeric import U256, Uint

from ethereum.crypto.hash import keccak256

from .fork_types import EMPTY_ACCOUNT, Account, Address, Root
from .trie import (
    EMPTY_TRIE_ROOT,
    IncrementalMPT,
    Trie,
    Witness,
    build_mpt,
    copy_trie,
    mpt_get,
    mpt_root,
    mpt_set,
    root,
    trie_get,
    trie_set,
)


@dataclass
class WitnessState:
    """
    Tracks state for deferred execution witness generation.

    Instead of recording witness nodes during execution, we preserve the
    pre-block state and track which keys are accessed (reads) and modified
    (writes). The witness is generated after all block execution by building
    a fresh IncrementalMPT from the pre-block state, traversing read paths,
    and applying the final diff for writes.
    """

    # Pre-block state (preserved at enable_witness_mode time)
    pre_state_accounts: Dict[Address, Optional[Account]]
    pre_state_storages: Dict[Address, Dict[Bytes32, U256]]

    # Dirty tracking during execution (writes)
    dirty_accounts: Set[Address] = field(default_factory=set)
    dirty_storage: Dict[Address, Set[Bytes32]] = field(default_factory=dict)

    # Access tracking during execution (reads)
    accessed_accounts: Set[Address] = field(default_factory=set)
    accessed_storage: Dict[Address, Set[Bytes32]] = field(default_factory=dict)

    # Bytecode tracking (code_hash -> bytecode)
    accessed_bytecodes: Dict[Bytes32, Bytes] = field(default_factory=dict)

    # Ancestor tracking - oldest block accessed via BLOCKHASH
    # All headers from this block to parent are needed for chain validation
    oldest_accessed_block: Optional[Uint] = None

    # Block metadata for ancestor collection
    current_block_number: Uint = field(default_factory=lambda: Uint(0))
    block_headers: List[Bytes] = field(default_factory=list)

    # Pre-execution MPTs
    _main_mpt: Optional["IncrementalMPT"] = None
    _storage_mpts: Optional[Dict[Address, "IncrementalMPT"]] = None


@dataclass
class State:
    """
    Contains all information that is preserved between transactions.
    """

    _main_trie: Trie[Address, Optional[Account]] = field(
        default_factory=lambda: Trie(secured=True, default=None)
    )
    _storage_tries: Dict[Address, Trie[Bytes32, U256]] = field(
        default_factory=dict
    )
    _snapshots: List[
        Tuple[
            Trie[Address, Optional[Account]],
            Dict[Address, Trie[Bytes32, U256]],
        ]
    ] = field(default_factory=list)
    created_accounts: Set[Address] = field(default_factory=set)
    _witness_state: Optional[WitnessState] = None


@dataclass
class TransientStorage:
    """
    Contains all information that is preserved between message calls
    within a transaction.
    """

    _tries: Dict[Address, Trie[Bytes32, U256]] = field(default_factory=dict)
    _snapshots: List[Dict[Address, Trie[Bytes32, U256]]] = field(
        default_factory=list
    )


def close_state(state: State) -> None:
    """
    Free resources held by the state. Used by optimized implementations to
    release file descriptors.
    """
    del state._main_trie
    del state._storage_tries
    del state._snapshots
    del state.created_accounts
    del state._witness_state


def begin_transaction(
    state: State, transient_storage: TransientStorage
) -> None:
    """
    Start a state transaction.

    Transactions are entirely implicit and can be nested. It is not possible to
    calculate the state root during a transaction.

    Parameters
    ----------
    state : State
        The state.
    transient_storage : TransientStorage
        The transient storage of the transaction.

    """
    state._snapshots.append(
        (
            copy_trie(state._main_trie),
            {k: copy_trie(t) for (k, t) in state._storage_tries.items()},
        )
    )
    transient_storage._snapshots.append(
        {k: copy_trie(t) for (k, t) in transient_storage._tries.items()}
    )


def commit_transaction(
    state: State, transient_storage: TransientStorage
) -> None:
    """
    Commit a state transaction.

    Parameters
    ----------
    state : State
        The state.
    transient_storage : TransientStorage
        The transient storage of the transaction.

    """
    state._snapshots.pop()
    if not state._snapshots:
        state.created_accounts.clear()

    transient_storage._snapshots.pop()


def rollback_transaction(
    state: State, transient_storage: TransientStorage
) -> None:
    """
    Rollback a state transaction, resetting the state to the point when the
    corresponding `begin_transaction()` call was made.

    Parameters
    ----------
    state : State
        The state.
    transient_storage : TransientStorage
        The transient storage of the transaction.

    """
    state._main_trie, state._storage_tries = state._snapshots.pop()
    if not state._snapshots:
        state.created_accounts.clear()

    transient_storage._tries = transient_storage._snapshots.pop()


def get_account(state: State, address: Address) -> Account:
    """
    Get the `Account` object at an address. Returns `EMPTY_ACCOUNT` if there
    is no account at the address.

    Use `get_account_optional()` if you care about the difference between a
    non-existent account and `EMPTY_ACCOUNT`.

    Parameters
    ----------
    state: `State`
        The state
    address : `Address`
        Address to lookup.

    Returns
    -------
    account : `Account`
        Account at address.

    """
    account = get_account_optional(state, address)
    if isinstance(account, Account):
        return account
    else:
        return EMPTY_ACCOUNT


def get_account_optional(state: State, address: Address) -> Optional[Account]:
    """
    Get the `Account` object at an address. Returns `None` (rather than
    `EMPTY_ACCOUNT`) if there is no account at the address.

    Parameters
    ----------
    state: `State`
        The state
    address : `Address`
        Address to lookup.

    Returns
    -------
    account : `Account`
        Account at address.

    """
    # Track accessed account for execution witness generation
    if state._witness_state is not None:
        state._witness_state.accessed_accounts.add(address)

    return trie_get(state._main_trie, address)


def set_account(
    state: State, address: Address, account: Optional[Account]
) -> None:
    """
    Set the `Account` object at an address. Setting to `None` deletes
    the account (but not its storage, see `destroy_account()`).

    Parameters
    ----------
    state: `State`
        The state
    address : `Address`
        Address to set.
    account : `Account`
        Account to set at address.

    """
    trie_set(state._main_trie, address, account)

    # Track dirty account for deferred witness generation
    if state._witness_state is not None:
        state._witness_state.dirty_accounts.add(address)


def destroy_account(state: State, address: Address) -> None:
    """
    Completely remove the account at `address` and all of its storage.

    This function is made available exclusively for the `SELFDESTRUCT`
    opcode. It is expected that `SELFDESTRUCT` will be disabled in a future
    hardfork and this function will be removed.

    Parameters
    ----------
    state: `State`
        The state
    address : `Address`
        Address of account to destroy.

    """
    destroy_storage(state, address)
    set_account(state, address, None)


def track_bytecode_access(state: State, code: Bytes) -> None:
    """
    Track bytecode access for execution witness generation.

    Should be called when bytecode is accessed for execution purposes
    (CALL variants, EXTCODESIZE, EXTCODECOPY, system contracts).

    Parameters
    ----------
    state : State
        The state with optional witness tracking.
    code : Bytes
        The bytecode being accessed.

    """
    # Skip if witness mode disabled or empty bytecode (EOAs)
    if state._witness_state is None or len(code) == 0:
        return

    # Compute hash and store for deduplication
    code_hash = Bytes32(keccak256(code))
    if code_hash not in state._witness_state.accessed_bytecodes:
        state._witness_state.accessed_bytecodes[code_hash] = code


def track_block_hash_access(state: State, block_number: Uint) -> None:
    """
    Track a block hash access for execution witness generation.

    Called when BLOCKHASH opcode or system contracts access a block hash.
    Tracks the oldest block accessed since all headers from that block
    to the parent are needed for chain validation.

    Parameters
    ----------
    state : State
        The state with optional witness tracking.
    block_number : Uint
        The block number being accessed.

    """
    if state._witness_state is None:
        return

    ws = state._witness_state
    is_oldest = (
        ws.oldest_accessed_block is None
        or block_number < ws.oldest_accessed_block
    )
    if is_oldest:
        ws.oldest_accessed_block = block_number


def set_witness_metadata(
    state: State, current_block_number: Uint, block_headers: List[Bytes]
) -> None:
    """
    Set block metadata needed for ancestor collection in witness generation.

    Parameters
    ----------
    state : State
        The state with witness tracking enabled.
    current_block_number : Uint
        The current block number being executed.
    block_headers : List[Bytes]
        RLP-encoded headers of previous blocks (up to 256).

    """
    if state._witness_state is None:
        return

    state._witness_state.current_block_number = current_block_number
    state._witness_state.block_headers = block_headers


def destroy_storage(state: State, address: Address) -> None:
    """
    Completely remove the storage at `address`.

    Parameters
    ----------
    state: `State`
        The state
    address : `Address`
        Address of account whose storage is to be deleted.

    """
    # Track all pre-block storage keys as dirty for witness generation
    if state._witness_state is not None:
        ws = state._witness_state
        if address in ws.pre_state_storages:
            # Mark all pre-block storage keys as dirty (they're now deleted)
            ws.dirty_storage.setdefault(address, set()).update(
                ws.pre_state_storages[address].keys()
            )

    if address in state._storage_tries:
        del state._storage_tries[address]


def mark_account_created(state: State, address: Address) -> None:
    """
    Mark an account as having been created in the current transaction.
    This information is used by `get_storage_original()` to handle an obscure
    edgecase, and to respect the constraints added to SELFDESTRUCT by
    EIP-6780.

    The marker is not removed even if the account creation reverts. Since the
    account cannot have had code prior to its creation and can't call
    `get_storage_original()`, this is harmless.

    Parameters
    ----------
    state: `State`
        The state
    address : `Address`
        Address of the account that has been created.

    """
    state.created_accounts.add(address)


def get_storage(state: State, address: Address, key: Bytes32) -> U256:
    """
    Get a value at a storage key on an account. Returns `U256(0)` if the
    storage key has not been set previously.

    Parameters
    ----------
    state: `State`
        The state
    address : `Address`
        Address of the account.
    key : `Bytes`
        Key to lookup.

    Returns
    -------
    value : `U256`
        Value at the key.

    """
    # Track accessed storage for execution witness generation
    if state._witness_state is not None:
        ws = state._witness_state
        ws.accessed_storage.setdefault(address, set()).add(key)

    trie = state._storage_tries.get(address)
    if trie is None:
        return U256(0)
    value = trie_get(trie, key)
    assert isinstance(value, U256)
    return value


def set_storage(
    state: State, address: Address, key: Bytes32, value: U256
) -> None:
    """
    Set a value at a storage key on an account. Setting to `U256(0)` deletes
    the key.

    Parameters
    ----------
    state: `State`
        The state
    address : `Address`
        Address of the account.
    key : `Bytes`
        Key to set.
    value : `U256`
        Value to set at the key.

    """
    assert trie_get(state._main_trie, address) is not None

    trie = state._storage_tries.get(address)
    if trie is None:
        trie = Trie(secured=True, default=U256(0))
        state._storage_tries[address] = trie
    trie_set(trie, key, value)
    if trie._data == {}:
        del state._storage_tries[address]

    # Track dirty storage for witness generation
    if state._witness_state is not None:
        state._witness_state.dirty_storage.setdefault(address, set()).add(key)


def storage_root(state: State, address: Address) -> Root:
    """
    Calculate the storage root of an account.

    Parameters
    ----------
    state:
        The state
    address :
        Address of the account.

    Returns
    -------
    root : `Root`
        Storage root of the account.

    """
    assert not state._snapshots
    if address in state._storage_tries:
        return root(state._storage_tries[address])
    else:
        return EMPTY_TRIE_ROOT


def state_root(state: State) -> Root:
    """
    Calculate the state root.

    Parameters
    ----------
    state:
        The current state.

    Returns
    -------
    root : `Root`
        The state root.

    """
    assert not state._snapshots

    def get_storage_root(address: Address) -> Root:
        return storage_root(state, address)

    # Calculate root using patricialize (existing implementation)
    patricialize_root = root(
        state._main_trie, get_storage_root=get_storage_root
    )

    # If witness mode is enabled, verify IncrementalMPT produces same root
    if state._witness_state is not None:
        inc_root = incremental_state_root(state)
        assert patricialize_root == inc_root, (
            f"Root mismatch! patricialize={patricialize_root.hex()} "
            f"incremental={inc_root.hex()}"
        )

    return patricialize_root


def account_exists(state: State, address: Address) -> bool:
    """
    Checks if an account exists in the state trie.

    Parameters
    ----------
    state:
        The state
    address:
        Address of the account that needs to be checked.

    Returns
    -------
    account_exists : `bool`
        True if account exists in the state trie, False otherwise

    """
    return get_account_optional(state, address) is not None


def account_has_code_or_nonce(state: State, address: Address) -> bool:
    """
    Checks if an account has non-zero nonce or non-empty code.

    Parameters
    ----------
    state:
        The state
    address:
        Address of the account that needs to be checked.

    Returns
    -------
    has_code_or_nonce : `bool`
        True if the account has non-zero nonce or non-empty code,
        False otherwise.

    """
    account = get_account(state, address)
    return account.nonce != Uint(0) or account.code != b""


def account_has_storage(state: State, address: Address) -> bool:
    """
    Checks if an account has storage.

    Parameters
    ----------
    state:
        The state
    address:
        Address of the account that needs to be checked.

    Returns
    -------
    has_storage : `bool`
        True if the account has storage, False otherwise.

    """
    return address in state._storage_tries


def is_account_alive(state: State, address: Address) -> bool:
    """
    Check whether an account is both in the state and non-empty.

    Parameters
    ----------
    state:
        The state
    address:
        Address of the account that needs to be checked.

    Returns
    -------
    is_alive : `bool`
        True if the account is alive.

    """
    account = get_account_optional(state, address)
    return account is not None and account != EMPTY_ACCOUNT


def modify_state(
    state: State, address: Address, f: Callable[[Account], None]
) -> None:
    """
    Modify an `Account` in the `State`. If, after modification, the account
    exists and has zero nonce, empty code, and zero balance, it is destroyed.
    """
    set_account(state, address, modify(get_account(state, address), f))

    account = get_account_optional(state, address)
    account_exists_and_is_empty = (
        account is not None
        and account.nonce == Uint(0)
        and account.code == b""
        and account.balance == 0
    )

    if account_exists_and_is_empty:
        destroy_account(state, address)


def move_ether(
    state: State,
    sender_address: Address,
    recipient_address: Address,
    amount: U256,
) -> None:
    """
    Move funds between accounts.
    """

    def reduce_sender_balance(sender: Account) -> None:
        if sender.balance < amount:
            raise AssertionError
        sender.balance -= amount

    def increase_recipient_balance(recipient: Account) -> None:
        recipient.balance += amount

    modify_state(state, sender_address, reduce_sender_balance)
    modify_state(state, recipient_address, increase_recipient_balance)


def set_account_balance(state: State, address: Address, amount: U256) -> None:
    """
    Sets the balance of an account.

    Parameters
    ----------
    state:
        The current state.

    address:
        Address of the account whose nonce needs to be incremented.

    amount:
        The amount that needs to be set in the balance.

    """

    def set_balance(account: Account) -> None:
        account.balance = amount

    modify_state(state, address, set_balance)


def increment_nonce(state: State, address: Address) -> None:
    """
    Increments the nonce of an account.

    Parameters
    ----------
    state:
        The current state.

    address:
        Address of the account whose nonce needs to be incremented.

    """

    def increase_nonce(sender: Account) -> None:
        sender.nonce += Uint(1)

    modify_state(state, address, increase_nonce)


def set_code(state: State, address: Address, code: Bytes) -> None:
    """
    Sets Account code.

    Parameters
    ----------
    state:
        The current state.

    address:
        Address of the account whose code needs to be updated.

    code:
        The bytecode that needs to be set.

    """

    def write_code(sender: Account) -> None:
        sender.code = code

    modify_state(state, address, write_code)


def get_storage_original(state: State, address: Address, key: Bytes32) -> U256:
    """
    Get the original value in a storage slot i.e. the value before the current
    transaction began. This function reads the value from the snapshots taken
    before executing the transaction.

    Parameters
    ----------
    state:
        The current state.
    address:
        Address of the account to read the value from.
    key:
        Key of the storage slot.

    """
    # In the transaction where an account is created, its preexisting storage
    # is ignored.
    if address in state.created_accounts:
        return U256(0)

    _, original_trie = state._snapshots[0]
    original_account_trie = original_trie.get(address)

    if original_account_trie is None:
        original_value = U256(0)
    else:
        original_value = trie_get(original_account_trie, key)

    assert isinstance(original_value, U256)

    return original_value


def get_transient_storage(
    transient_storage: TransientStorage, address: Address, key: Bytes32
) -> U256:
    """
    Get a value at a storage key on an account from transient storage.
    Returns `U256(0)` if the storage key has not been set previously.

    Parameters
    ----------
    transient_storage: `TransientStorage`
        The transient storage
    address : `Address`
        Address of the account.
    key : `Bytes`
        Key to lookup.

    Returns
    -------
    value : `U256`
        Value at the key.

    """
    trie = transient_storage._tries.get(address)
    if trie is None:
        return U256(0)

    value = trie_get(trie, key)

    assert isinstance(value, U256)
    return value


def set_transient_storage(
    transient_storage: TransientStorage,
    address: Address,
    key: Bytes32,
    value: U256,
) -> None:
    """
    Set a value at a storage key on an account. Setting to `U256(0)` deletes
    the key.

    Parameters
    ----------
    transient_storage: `TransientStorage`
        The transient storage
    address : `Address`
        Address of the account.
    key : `Bytes`
        Key to set.
    value : `U256`
        Value to set at the key.

    """
    trie = transient_storage._tries.get(address)
    if trie is None:
        trie = Trie(secured=True, default=U256(0))
        transient_storage._tries[address] = trie
    trie_set(trie, key, value)
    if trie._data == {}:
        del transient_storage._tries[address]


def enable_witness_mode(state: State) -> None:
    """
    Enable witness tracking mode for the state.

    Preserves the current (pre-block) state and sets up dirty tracking.
    The actual witness generation is deferred until after block execution.

    Parameters
    ----------
    state :
        The state to enable witness mode on.

    """
    assert not state._snapshots, "Cannot enable witness during transaction"

    state._witness_state = WitnessState(
        pre_state_accounts=dict(state._main_trie._data),
        pre_state_storages={
            addr: dict(trie._data)
            for addr, trie in state._storage_tries.items()
        },
    )


def is_witness_mode_enabled(state: State) -> bool:
    """
    Check if witness tracking mode is enabled.

    Parameters
    ----------
    state :
        The state to check.

    Returns
    -------
    enabled : `bool`
        True if witness mode is enabled.

    """
    return state._witness_state is not None


def _build_witness_mpts(state: State) -> None:
    """
    Build and cache the IncrementalMPTs for witness generation.

    This builds the MPTs from pre-block state, applies all reads and writes,
    which records witness nodes as a side effect. The MPTs are cached in
    WitnessState for reuse by root computation and witness extraction.

    Parameters
    ----------
    state :
        The state with witness tracking enabled.

    """
    assert state._witness_state is not None
    ws = state._witness_state

    # Already built
    if ws._main_mpt is not None:
        return

    # Build pre-block storage MPTs
    storage_mpts: Dict[Address, IncrementalMPT[Bytes32, U256]] = {}
    for address, data in ws.pre_state_storages.items():
        storage_mpts[address] = build_mpt(
            dict(data), secured=True, default=U256(0)
        )

    def get_pre_storage_root(address: Address) -> Root:
        if address in storage_mpts:
            return mpt_root(storage_mpts[address])
        return EMPTY_TRIE_ROOT

    main_mpt = build_mpt(
        dict(ws.pre_state_accounts),
        secured=True,
        default=None,
        get_storage_root=get_pre_storage_root,
    )

    # 1. Do read-only storages accesses
    for address, accessed_keys in ws.accessed_storage.items():
        if address not in storage_mpts:
            continue

        for key in accessed_keys:
            mpt_get(storage_mpts[address], key)

    # 2. Apply dirty storage to storages (writes)
    for address, dirty_keys in ws.dirty_storage.items():
        if address not in storage_mpts:
            # New storage created during block
            storage_mpts[address] = build_mpt(
                {}, secured=True, default=U256(0)
            )

        storage_trie = state._storage_tries.get(address)
        # We do two passes to ensure deletions are processed after
        # inserts/updates to minimize the number of nodes touched
        # in the MPT.
        # First pass: inserts and updates
        for key in dirty_keys:
            value = trie_get(storage_trie, key) if storage_trie else U256(0)
            if value != 0:
                mpt_set(storage_mpts[address], key, value)
        # Second pass: deletions
        for key in dirty_keys:
            value = trie_get(storage_trie, key) if storage_trie else U256(0)
            if value == 0:
                mpt_set(storage_mpts[address], key, value)

    # Accounts are "dirty" if:
    # - Account fields changed (nonce/balance/code) - tracked in dirty_accounts
    # - Storage changed (storage root changed) - tracked in dirty_storage
    all_dirty_accounts = ws.dirty_accounts | set(ws.dirty_storage.keys())

    # 3. Traverse accounts that were read
    for address in ws.accessed_accounts:
        mpt_get(main_mpt, address)

    # 4. Apply dirty accounts
    for address in all_dirty_accounts:
        # Get new account data from usual trie
        account = trie_get(state._main_trie, address)

        # Get storage root for this account
        if address in storage_mpts:
            addr_storage_root = mpt_root(storage_mpts[address])
            # Verify invariant: MPT root must match state storage root
            # (if storage was fully cleared, it won't be in
            # state._storage_tries)
            if address in state._storage_tries:
                assert addr_storage_root == root(state._storage_tries[address])
            else:
                assert addr_storage_root == EMPTY_TRIE_ROOT
        else:
            # Verify invariant: no storage in state either
            assert address not in state._storage_tries
            addr_storage_root = EMPTY_TRIE_ROOT

        def get_storage_root_fn(
            _: Address, sr: Root = addr_storage_root
        ) -> Root:
            return sr

        mpt_set(
            main_mpt,
            address,
            account,
            get_storage_root=get_storage_root_fn,
        )

    # Cache the built MPTs
    ws._main_mpt = main_mpt
    ws._storage_mpts = storage_mpts


def incremental_state_root(state: State) -> Root:
    """
    Compute state root using IncrementalMPT.

    Builds the MPTs if not already built, then returns the root.
    The MPT nodes cache their hashes, so subsequent calls are fast.

    Parameters
    ----------
    state :
        The state with witness tracking enabled.

    Returns
    -------
    root : `Root`
        The state root computed via IncrementalMPT.

    """
    assert state._witness_state is not None
    _build_witness_mpts(state)
    assert state._witness_state._main_mpt is not None
    assert state._witness_state._storage_mpts is not None
    return mpt_root(state._witness_state._main_mpt)


def generate_witness(state: State) -> Tuple[Root, Witness]:
    """
    Generate execution witness.

    Parameters
    ----------
    state :
        The state with witness tracking enabled.

    Returns
    -------
    root : `Root`
        The state root computed via IncrementalMPT.
    witness : `Witness`
        The execution witness containing accessed nodes.

    """
    assert state._witness_state is not None
    ws = state._witness_state

    # Ensure MPTs are built
    _build_witness_mpts(state)
    assert ws._main_mpt is not None
    assert ws._storage_mpts is not None

    main_mpt = ws._main_mpt
    storage_mpts = ws._storage_mpts

    # Collect ancestors from parent (newest) to oldest accessed block
    # All headers in this range needed for parent hash chain validation
    # Order: [parent, grandparent, ..., oldest] per EIP spec
    ancestors: List[Bytes] = []
    assert ws.oldest_accessed_block is not None and ws.block_headers
    # Include all headers from parent down to oldest accessed block
    for block_num in range(
        int(ws.current_block_number) - 1, int(ws.oldest_accessed_block) - 1, -1
    ):
        offset = int(ws.current_block_number) - block_num
        if offset <= len(ws.block_headers):
            header_rlp = ws.block_headers[-offset]
            if header_rlp:
                ancestors.append(header_rlp)

    # Collect witness from all MPTs
    witness = Witness(
        accessed_nodes=dict(main_mpt.witness.accessed_nodes),
        accessed_keys=set(main_mpt.witness.accessed_keys),
        bytecodes=sorted(ws.accessed_bytecodes.values()),
        ancestors=ancestors,
    )
    for mpt in storage_mpts.values():
        witness.accessed_nodes.update(mpt.witness.accessed_nodes)
        witness.accessed_keys.update(mpt.witness.accessed_keys)

    return mpt_root(main_mpt), witness


def get_witness(state: State) -> Witness:
    """
    Get the collected witness data from the state.

    Parameters
    ----------
    state :
        The state with witness tracking enabled.

    Returns
    -------
    witness : `Witness`
        The witness data containing accessed nodes.

    """
    if state._witness_state is None:
        return Witness()

    _, witness = generate_witness(state)
    return witness
