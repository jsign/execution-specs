"""
Stateless Execution.

.. contents:: Table of Contents
    :backlinks: none
    :local:

Introduction
------------

This module provides the entry point for stateless block validation,
where a block is validated using only an execution witness (containing
pre-state trie nodes, bytecodes, and ancestor headers) rather than
full state access.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union

from ethereum_rlp import rlp
from ethereum_types.bytes import Bytes, Bytes8, Bytes32
from ethereum_types.numeric import U64, U256, Uint

from ethereum.crypto.hash import Hash32, keccak256
from ethereum.exceptions import InvalidBlock

from .blocks import (
    Block,
    ExecutionWitness,
    Header,
    NewPayloadRequest,
    StatelessInput,
    Withdrawal,
)
from .fork import EMPTY_OMMER_HASH, state_transition
from .fork_types import (
    Address,
    Bloom,
    Root,
    VersionedHash,
)
from .requests import compute_requests_hash
from .transactions import (
    BlobTransaction,
    LegacyTransaction,
    decode_transaction,
)
from .trie import (
    EMPTY_TRIE_ROOT,
    Trie,
    WitnessBackedTrie,
    build_witness_trie,
    copy_witness_trie,
    root,
    trie_set,
    witness_trie_get,
    witness_trie_root,
    witness_trie_set,
)


class _Deleted:
    """
    Sentinel value to mark an entry as explicitly deleted.

    Used in the diff layer to distinguish between:
    - Key not in cache (need to fetch from base layer)
    - Key explicitly deleted (return None/zero)
    - Key exists with a value
    """

    pass


DELETED = _Deleted()
"""Singleton sentinel for deleted entries."""


EMPTY_CODE_HASH = Hash32(keccak256(b""))
"""Hash of empty bytecode."""


@dataclass
class AccountData:
    """
    Account data as stored in the witness trie (without actual bytecode).

    The witness trie stores accounts as RLP(nonce, balance, storage_root, code_hash).
    We store nonce, balance, code_hash here. Storage root is derived from storage trie.

    This separates account data from bytecode, allowing witnesses that only
    include bytecode when code is actually executed.
    """

    nonce: Uint
    balance: U256
    code_hash: Hash32


@dataclass
class WitnessBaseLayer:
    """
    Immutable pre-state layer built from execution witness.

    This layer is NEVER modified during EVM execution. It provides
    pre-state data for accounts, storage, and bytecode from the witness.

    Queries that hit missing witness data (StubNode) will raise
    InvalidBlock - we do NOT return default values for missing data.
    """

    _main_trie: WitnessBackedTrie
    """Read-only state trie built from witness nodes."""

    _node_map: Dict[Hash32, Bytes] = field(default_factory=dict)
    """Mapping from node hash to RLP-encoded node (for building storage tries)."""

    _bytecode_map: Dict[Hash32, Bytes] = field(default_factory=dict)
    """Mapping from code hash to bytecode."""

    _storage_tries_cache: Dict[Address, WitnessBackedTrie] = field(
        default_factory=dict
    )
    """Lazily built storage tries from witness (read-only after built)."""


@dataclass
class DiffLayer:
    """
    Mutable cache/diff layer for EVM execution.

    EVM execution accesses this layer first. If data is not present,
    it's fetched from the base layer and cached here. All modifications
    are written to this layer only.

    Uses DELETED sentinel to distinguish "deleted" from "not cached".
    Separates account data from bytecode for lazy bytecode loading.
    """

    _accounts: Dict[Address, Union[AccountData, _Deleted]] = field(
        default_factory=dict
    )
    """
    Account data cache. Maps address to:
    - AccountData: account exists (nonce, balance, code_hash)
    - DELETED: account was explicitly deleted
    Key not in dict means not yet fetched from base layer.
    """

    _storage: Dict[Address, Dict[Bytes32, U256]] = field(default_factory=dict)
    """
    Storage cache. Maps address -> (slot -> value).
    U256(0) means the slot is empty/deleted (same semantics in Ethereum).
    Key not in inner dict means not yet fetched from base layer.
    """

    _bytecodes: Dict[Hash32, Bytes] = field(default_factory=dict)
    """
    Bytecode cache. Maps code_hash -> bytecode.
    Lazily populated when bytecode is actually needed (CALL, EXTCODESIZE, etc.).
    New bytecodes from CREATE/CREATE2 are also stored here.
    """

    _dirty_accounts: Set[Address] = field(default_factory=set)
    """Accounts that were modified (need to be included in state root)."""

    _dirty_storage: Dict[Address, Set[Bytes32]] = field(default_factory=dict)
    """Storage slots that were modified per account."""

    created_accounts: Set[Address] = field(default_factory=set)
    """Accounts created in current transaction (for EIP-6780 SELFDESTRUCT)."""


@dataclass
class WitnessBackedState:
    """
    Two-layer state implementation for stateless execution.

    Layer 1 (Base): Immutable witness data (pre-state)
    Layer 2 (Diff): Mutable cache/modifications (runtime state)

    The EVM accesses state through this class, which implements
    a two-layer lookup pattern:
    1. Check diff layer first
    2. If not present, fetch from base layer and cache
    3. Write operations only modify the diff layer

    At block end, state root is computed by applying diffs to
    trie copies in the correct order (inserts/updates first,
    deletions after) to avoid branch compression issues.
    """

    _base: WitnessBaseLayer
    """Immutable pre-state from witness."""

    _diff: DiffLayer
    """Mutable cache and modifications."""

    _snapshots: List[DiffLayer] = field(default_factory=list)
    """Stack of diff layer snapshots for transaction rollback."""


@dataclass
class WitnessBackedBlockChain:
    """
    BlockChain backed by witness ancestors for stateless execution.

    Instead of storing full Block objects, this derives block hashes from
    the witness ancestors list (RLP-encoded headers).
    """

    _ancestors: List[Bytes]  # RLP-encoded ancestor headers
    _ancestor_hashes: List[Hash32]  # Computed hashes (parent first)
    state: WitnessBackedState
    chain_id: U64


def _decode_account_data_from_witness(
    encoded: Bytes,
) -> Tuple[AccountData, Root]:
    """
    Decode account data from witness trie leaf value.

    The witness trie stores accounts as RLP(nonce, balance, storage_root,
    code_hash). This function extracts just the account data WITHOUT looking
    up the actual bytecode - bytecode is fetched lazily when needed.

    Parameters
    ----------
    encoded :
        RLP-encoded account data from witness trie.

    Returns
    -------
    account_data : AccountData
        The decoded account data (nonce, balance, code_hash).
    storage_root : Root
        The storage root for building the storage trie.

    """
    decoded = rlp.decode(encoded)
    nonce = Uint(decoded[0])
    balance = U256(decoded[1])
    storage_root = Root(decoded[2])
    code_hash = Hash32(decoded[3])

    account_data = AccountData(nonce=nonce, balance=balance, code_hash=code_hash)
    return account_data, storage_root


def _get_storage_root_from_encoded_account(encoded: Bytes) -> Root:
    """Extract storage root from RLP-encoded account without full decode."""
    decoded = rlp.decode(encoded)
    return Root(decoded[2])


EMPTY_ACCOUNT_DATA = AccountData(
    nonce=Uint(0),
    balance=U256(0),
    code_hash=EMPTY_CODE_HASH,
)
"""Default data for non-existent accounts."""


def witness_get_account_optional(
    state: WitnessBackedState,
    address: Address,
) -> Optional[AccountData]:
    """
    Get account data from witness-backed state using two-layer lookup.

    1. Check diff layer first
    2. If not in diff, fetch from base layer and cache
    3. Return None if account doesn't exist (proven by witness structure)

    Does NOT fetch bytecode - use witness_get_code() when bytecode is needed.
    Raises InvalidBlock if witness is incomplete (hits StubNode).

    Parameters
    ----------
    state :
        The witness-backed state.
    address :
        Address to lookup.

    Returns
    -------
    account_data : Optional[AccountData]
        The account data at address, or None if it doesn't exist.

    """
    if address in state._diff._accounts:
        value = state._diff._accounts[address]
        if isinstance(value, _Deleted):
            return None
        return value

    encoded = witness_trie_get(state._base._main_trie, address)

    if encoded is None:
        return None

    account_data, _ = _decode_account_data_from_witness(encoded)
    state._diff._accounts[address] = account_data
    return account_data


def witness_get_account(
    state: WitnessBackedState, address: Address
) -> AccountData:
    """Get account data, or EMPTY_ACCOUNT_DATA if not found."""
    account_data = witness_get_account_optional(state, address)
    if account_data is None:
        return EMPTY_ACCOUNT_DATA
    return account_data


def witness_get_code(
    state: WitnessBackedState,
    address: Address,
) -> Bytes:
    """
    Get bytecode for an account using lazy two-layer lookup.

    Bytecode is fetched only when actually needed (CALL, EXTCODESIZE, etc.),
    not when account data is retrieved.

    1. Get account data to find code_hash
    2. If empty code hash -> return b""
    3. Check diff layer bytecode cache first
    4. If not cached -> look up from base layer bytecode_map
    5. If not in bytecode_map -> fail with InvalidBlock (incomplete witness)

    Parameters
    ----------
    state :
        The witness-backed state.
    address :
        Address of the account.

    Returns
    -------
    code : Bytes
        The bytecode, or b"" if empty.

    Raises
    ------
    InvalidBlock
        If bytecode is needed but not present in witness.

    """
    account_data = witness_get_account_optional(state, address)
    if account_data is None:
        return b""  # Non-existent account has no code

    code_hash = account_data.code_hash

    # Empty code
    if code_hash == EMPTY_CODE_HASH:
        return b""

    # Check diff layer cache first
    if code_hash in state._diff._bytecodes:
        return state._diff._bytecodes[code_hash]

    # Look up from base layer
    if code_hash in state._base._bytecode_map:
        code = state._base._bytecode_map[code_hash]
        # Cache in diff layer
        state._diff._bytecodes[code_hash] = code
        return code

    # Bytecode not in witness - incomplete witness for this operation
    raise InvalidBlock(
        f"Bytecode for code_hash {code_hash.hex()} not in witness. "
        "Witness is incomplete for operations requiring this bytecode."
    )


def witness_set_account(
    state: WitnessBackedState,
    address: Address,
    account_data: Optional[AccountData],
) -> None:
    """
    Set account data in witness-backed state (diff layer only).

    Does NOT modify the base layer. All modifications go to the diff layer
    and are applied during state root computation.

    For new accounts with code (CREATE/CREATE2), the caller should also
    call witness_set_code() to store the bytecode.

    Parameters
    ----------
    state :
        The witness-backed state.
    address :
        Address to set.
    account_data :
        Account data to set, or None to delete.

    """
    if account_data is None:
        # Mark account as deleted in diff layer
        state._diff._accounts[address] = DELETED
        state._diff._storage.pop(address, None)
        state._diff._dirty_storage.pop(address, None)
    else:
        state._diff._accounts[address] = account_data

    state._diff._dirty_accounts.add(address)


def witness_set_code(
    state: WitnessBackedState,
    code: Bytes,
) -> Hash32:
    """
    Store bytecode in the diff layer and return its hash.

    Used when deploying new contracts (CREATE/CREATE2) to store the
    bytecode for later retrieval.

    Parameters
    ----------
    state :
        The witness-backed state.
    code :
        The bytecode to store.

    Returns
    -------
    code_hash : Hash32
        The keccak256 hash of the bytecode.

    """
    code_hash = Hash32(keccak256(code))
    if code_hash != EMPTY_CODE_HASH:
        state._diff._bytecodes[code_hash] = code
    return code_hash


def _get_base_storage_trie(
    state: WitnessBackedState,
    address: Address,
) -> WitnessBackedTrie:
    """
    Get or lazily build a storage trie from the base (witness) layer.

    If the storage trie hasn't been built yet, it's constructed from
    the witness node_map using the account's storage_root.

    This trie is for READ-ONLY access from the base layer. It should
    NOT be modified during execution.

    Parameters
    ----------
    state :
        The witness-backed state.
    address :
        Address of the account.

    Returns
    -------
    storage_trie : WitnessBackedTrie
        The storage trie for the account from the base layer.

    """
    if address in state._base._storage_tries_cache:
        return state._base._storage_tries_cache[address]

    # Get account's storage root from base layer
    encoded = witness_trie_get(state._base._main_trie, address)
    if encoded is None:
        # Account doesn't exist in base layer - empty storage trie
        storage_trie = WitnessBackedTrie(root_node=None)
    else:
        storage_root = _get_storage_root_from_encoded_account(encoded)
        if storage_root == EMPTY_TRIE_ROOT:
            storage_trie = WitnessBackedTrie(root_node=None)
        else:
            storage_trie = build_witness_trie(
                state._base._node_map, storage_root
            )

    state._base._storage_tries_cache[address] = storage_trie
    return storage_trie


def witness_get_storage(
    state: WitnessBackedState,
    address: Address,
    key: Bytes32,
) -> U256:
    """
    Get a storage value from witness-backed state using two-layer lookup.

    1. Check diff layer first
    2. If not in diff, fetch from base layer and cache
    3. Return U256(0) if slot doesn't exist

    Parameters
    ----------
    state :
        The witness-backed state.
    address :
        Address of the account.
    key :
        Storage key.

    Returns
    -------
    value : U256
        Storage value, or U256(0) if not set.

    """
    # Check diff layer first
    if address in state._diff._storage:
        storage_cache = state._diff._storage[address]
        if key in storage_cache:
            return storage_cache[key]  # U256(0) means empty/deleted

    # Check if account is deleted in diff
    if address in state._diff._accounts:
        if isinstance(state._diff._accounts[address], _Deleted):
            return U256(0)  # Account deleted, storage is zero

    # Fetch from base layer
    storage_trie = _get_base_storage_trie(state, address)
    encoded = witness_trie_get(storage_trie, key)

    if encoded is None:
        value = U256(0)
    else:
        value = U256(rlp.decode(encoded))

    # Cache the value in diff layer
    state._diff._storage.setdefault(address, {})[key] = value
    return value


def witness_set_storage(
    state: WitnessBackedState,
    address: Address,
    key: Bytes32,
    value: U256,
) -> None:
    """
    Set a storage value in witness-backed state (diff layer only).

    Does NOT modify the base layer. All modifications go to the diff layer
    and are applied during state root computation.

    Parameters
    ----------
    state :
        The witness-backed state.
    address :
        Address of the account.
    key :
        Storage key.
    value :
        Storage value. U256(0) deletes the key.

    """
    state._diff._storage.setdefault(address, {})[key] = value
    state._diff._dirty_storage.setdefault(address, set()).add(key)


def _copy_diff_layer(diff: DiffLayer) -> DiffLayer:
    """
    Create a deep copy of the diff layer for snapshotting.

    AccountData, U256, and Bytes objects are immutable/frozen, so shallow
    copies of the inner dicts are sufficient.

    Parameters
    ----------
    diff :
        The diff layer to copy.

    Returns
    -------
    copy : DiffLayer
        A deep copy of the diff layer.

    """
    return DiffLayer(
        _accounts=dict(diff._accounts),
        _storage={addr: dict(slots) for addr, slots in diff._storage.items()},
        _bytecodes=dict(diff._bytecodes),
        _dirty_accounts=set(diff._dirty_accounts),
        _dirty_storage={
            addr: set(keys) for addr, keys in diff._dirty_storage.items()
        },
        created_accounts=set(diff.created_accounts),
    )


def witness_begin_transaction(
    state: WitnessBackedState,
    transient_storage: object,
) -> None:
    """
    Snapshot the diff layer before a nested call.

    Used for transaction/call depth handling - if the call reverts,
    we can restore the diff layer from the snapshot.

    Parameters
    ----------
    state :
        The witness-backed state.
    transient_storage :
        Transient storage (passed for API compatibility with State).

    """
    state._snapshots.append(_copy_diff_layer(state._diff))


def witness_commit_transaction(
    state: WitnessBackedState,
    transient_storage: object,
) -> None:
    """
    Discard the snapshot on successful return from a nested call.

    Parameters
    ----------
    state :
        The witness-backed state.
    transient_storage :
        Transient storage (passed for API compatibility with State).

    """
    state._snapshots.pop()
    if not state._snapshots:
        state._diff.created_accounts.clear()


def witness_rollback_transaction(
    state: WitnessBackedState,
    transient_storage: object,
) -> None:
    """
    Restore the diff layer from a snapshot on revert.

    Parameters
    ----------
    state :
        The witness-backed state.
    transient_storage :
        Transient storage (passed for API compatibility with State).

    """
    state._diff = state._snapshots.pop()
    if not state._snapshots:
        state._diff.created_accounts.clear()


def _get_base_storage_root(
    state: WitnessBackedState,
    address: Address,
) -> Root:
    """
    Get the storage root for an account from the base layer.

    Parameters
    ----------
    state :
        The witness-backed state.
    address :
        Address of the account.

    Returns
    -------
    root : Root
        The storage root from the base layer.

    """
    encoded = witness_trie_get(state._base._main_trie, address)
    if encoded is None:
        return EMPTY_TRIE_ROOT
    return _get_storage_root_from_encoded_account(encoded)


def witness_state_root(state: WitnessBackedState) -> Root:
    """
    Compute the state root by applying diffs to base layer tries.

    Parameters
    ----------
    state :
        The witness-backed state.

    Returns
    -------
    root : Root
        The computed state root.

    """
    # Build working copies of tries for root computation
    # to avoid modifying the actual base layer
    main_trie = copy_witness_trie(state._base._main_trie)
    storage_tries: Dict[Address, WitnessBackedTrie] = {}

    # Process dirty storage
    for address, dirty_keys in state._diff._dirty_storage.items():
        # Get or copy storage trie from base layer
        if address not in storage_tries:
            base_trie = _get_base_storage_trie(state, address)
            storage_tries[address] = copy_witness_trie(base_trie)

        storage_trie = storage_tries[address]
        storage_cache = state._diff._storage.get(address, {})

        # First pass: inserts and updates
        for key in dirty_keys:
            value = storage_cache.get(key, U256(0))
            if value != U256(0):
                witness_trie_set(storage_trie, key, rlp.encode(value))

        # Second pass: deletions (U256(0) means empty/deleted in Ethereum)
        for key in dirty_keys:
            value = storage_cache.get(key, U256(0))
            if value == U256(0):
                witness_trie_set(storage_trie, key, b"")  # Delete

    # Process dirty accounts
    for address in state._diff._dirty_accounts:
        account_data = state._diff._accounts.get(address)
        assert account_data is not None, "dirty account can't be non-existent"
        if not isinstance(account_data, _Deleted):
            # Get storage root for this account
            if address in storage_tries:
                storage_root = witness_trie_root(storage_tries[address])
            else:
                # If address had dirty storage, it would be in storage_tries
                assert address not in state._diff._dirty_storage
                storage_root = _get_base_storage_root(state, address)

            # Encode account using code_hash directly (no bytecode needed!)
            encoded = rlp.encode((
                account_data.nonce,
                account_data.balance,
                storage_root,
                account_data.code_hash,
            ))
            witness_trie_set(main_trie, address, encoded)

    # Note: No deletion pass needed. Since EIP-6780, SELFDESTRUCT only deletes
    # accounts created in the same transaction. Such accounts were never in the
    # pre-state trie, so there's nothing to delete from main_trie.

    return witness_trie_root(main_trie)


@dataclass
class StatelessOutput:
    """
    Output of stateless state transition execution.
    """

    new_payload_request_root: Hash32
    success: bool


def validate_execution_requests(requests: List[Bytes]) -> bool:
    """
    Validate execution requests per EIP-7685 / Prague spec.

    Requirements:
    - Each element must be longer than 1 byte
    - Elements must be ordered by request_type (first byte) in ascending order
    - Each request_type must be unique

    Parameters
    ----------
    requests :
        List of execution request bytes.

    Returns
    -------
    valid : bool
        True if the requests are valid.

    """
    prev_type = -1
    for req in requests:
        if len(req) <= 1:
            return False
        req_type = req[0]
        if req_type <= prev_type:
            return False
        prev_type = req_type
    return True


def validate_new_payload_params(
    new_payload_request: NewPayloadRequest,
) -> bool:
    """
    Validate blob versioned hashes and execution requests per Engine API spec.

    From engine_newPayloadV3 (Cancun):
    - Verify expected blob versioned hashes match actual hashes from txs

    From engine_newPayloadV4 (Prague):
    - Verify execution requests are properly formatted

    Parameters
    ----------
    new_payload_request :
        The new payload request to validate (must be V5 format).

    Returns
    -------
    valid : bool
        True if the parameters are valid.

    """
    execution_payload = new_payload_request[0]
    expected_blob_hashes = new_payload_request[1]
    execution_requests = new_payload_request[3]

    # Validate blob versioned hashes
    actual_blob_hashes: List[VersionedHash] = []
    for tx_bytes in execution_payload.transactions:
        tx = decode_transaction(tx_bytes)
        if isinstance(tx, BlobTransaction):
            actual_blob_hashes.extend(tx.blob_versioned_hashes)

    if list(expected_blob_hashes) != actual_blob_hashes:
        return False

    # Validate execution requests
    if not validate_execution_requests(list(execution_requests)):
        return False

    return True


def validate_list_ordering(items: List[Bytes]) -> bool:
    """
    Validate that a list of bytes is sorted in ascending lexicographic order
    and contains no duplicates.

    Parameters
    ----------
    items :
        List of bytes to validate.

    Returns
    -------
    valid : bool
        True if items are properly ordered and unique.

    """
    if len(items) == 0:
        return True

    prev_item = items[0]
    for i in range(1, len(items)):
        current_item = items[i]
        # Strictly greater than ensures both ordering and deduplication
        if current_item <= prev_item:
            return False
        prev_item = current_item

    return True


def validate_ancestors_chain(
    ancestors: List[Bytes], parent_hash: Hash32
) -> bool:
    """
    Validate ancestor block headers form a valid chain.

    Requirements:
    - Minimum 1 element (parent block), maximum 256 elements
    - First ancestor's hash must equal parent_hash
    - Each header's parent_hash must equal keccak256 of the next ancestor

    Parameters
    ----------
    ancestors :
        RLP-encoded ancestor block headers (from parent to oldest).
    parent_hash :
        Expected hash of the parent block.

    Returns
    -------
    valid : bool
        True if ancestor chain is valid.

    """
    # Check count constraints
    if len(ancestors) == 0 or len(ancestors) > 256:
        return False

    # First ancestor should be the immediate parent
    first_ancestor_hash = Hash32(keccak256(ancestors[0]))
    if first_ancestor_hash != parent_hash:
        return False

    # Verify chain of parent hashes
    for i in range(len(ancestors) - 1):
        header = rlp.decode_to(Header, ancestors[i])
        expected_parent_hash = Hash32(keccak256(ancestors[i + 1]))
        if header.parent_hash != expected_parent_hash:
            return False

    return True


def _verify_node_recursive(
    node_map: Dict[Hash32, Bytes],
    node_hash: Hash32,
    visited: Set[Hash32],
) -> bool:
    """
    Recursively verify a node and its children exist in the node map.

    Parameters
    ----------
    node_map :
        Mapping from node hash to node bytes.
    node_hash :
        Hash of the current node to verify.
    visited :
        Set of already visited node hashes (for cycle detection).

    Returns
    -------
    valid : bool
        True if node and all children are valid.

    """
    if node_hash in visited:
        return False  # Cycle detected
    visited.add(node_hash)

    if node_hash not in node_map:
        return False

    node_rlp = node_map[node_hash]

    try:
        decoded: Union[Bytes, List] = rlp.decode(node_rlp)

        if not isinstance(decoded, list):
            return False

        if len(decoded) == 17:
            # Branch node: 16 children + value
            for i in range(16):
                child = decoded[i]
                if isinstance(child, bytes) and len(child) == 32:
                    # Child is a hash reference - only verify if in witness
                    # (witness may contain partial trie proofs)
                    child_hash = Hash32(child)
                    if child_hash in node_map:
                        result = _verify_node_recursive(
                            node_map, child_hash, visited
                        )
                        if not result:
                            return False
                # Embedded nodes (RLP < 32 bytes) and empty are valid
        elif len(decoded) == 2:
            # Extension or Leaf node - check compact encoding prefix
            path = decoded[0]
            if not isinstance(path, bytes) or len(path) == 0:
                return False
            # Bit 0x20 in first nibble indicates leaf node
            is_leaf = (path[0] & 0x20) != 0
            if not is_leaf:
                # Extension node - second element is child reference
                child = decoded[1]
                if isinstance(child, bytes) and len(child) == 32:
                    child_hash = Hash32(child)
                    if child_hash in node_map:
                        result = _verify_node_recursive(
                            node_map, child_hash, visited
                        )
                        if not result:
                            return False
            # Leaf node - second element is value, no recursion needed
        else:
            return False

    except Exception:
        return False

    return True


def verify_trie(
    nodes: List[Bytes],
    expected_root: Root,
) -> bool:
    """
    Verify that provided nodes can reconstruct a trie with the expected root.

    Builds a hash->node mapping and traverses from the root, verifying that
    all referenced nodes are present and form a valid MPT structure.

    Parameters
    ----------
    nodes :
        List of RLP-encoded trie nodes.
    expected_root :
        The expected state root (from parent block).

    Returns
    -------
    valid : bool
        True if nodes form a valid trie with the expected root.

    """
    # Build node map: keccak256(node) -> node
    node_map: Dict[Hash32, Bytes] = {}
    for node in nodes:
        node_hash = Hash32(keccak256(node))
        node_map[node_hash] = node

    # Handle empty trie case
    if expected_root == EMPTY_TRIE_ROOT:
        return len(node_map) == 0

    # Root must be in node map
    if expected_root not in node_map:
        return False

    # Traverse and verify structure
    visited: Set[Hash32] = set()
    return _verify_node_recursive(node_map, expected_root, visited)


def validate_execution_witness(
    witness: ExecutionWitness,
    parent_hash: Hash32,
) -> bool:
    """
    Validate an execution witness for stateless execution.

    Validates that:
    1. Ancestors form a valid chain starting from the parent block
    2. Nodes are RLP-encoded, deduplicated, and sorted in ascending order
    3. Nodes can reconstruct the pre-state trie with parent_state_root
    4. Bytecodes are deduplicated and sorted in ascending order

    Parameters
    ----------
    witness :
        The ExecutionWitness containing nodes, bytecodes, and ancestors.
    parent_hash :
        The hash of the parent block.

    Returns
    -------
    valid : bool
        True if the execution witness is valid, False otherwise.

    """
    # Validate ancestors chain (non-empty, hash matches, chain integrity)
    if not validate_ancestors_chain(list(witness.ancestors), parent_hash):
        return False

    # Decode parent header to get state_root (safe after ancestors validation)
    try:
        parent_header = rlp.decode_to(Header, witness.ancestors[0])
    except Exception:
        return False
    parent_state_root = parent_header.state_root

    # Validate nodes ordering and deduplication
    if not validate_list_ordering(list(witness.nodes)):
        return False

    # Validate bytecodes ordering and deduplication
    if not validate_list_ordering(list(witness.bytecodes)):
        return False

    # Verify trie reconstruction from witness nodes
    if not verify_trie(list(witness.nodes), parent_state_root):
        return False

    return True


def block_from_new_payload_request(
    new_payload_request: NewPayloadRequest,
) -> Block:
    """
    Convert an EngineNewPayloadV5Parameters to a Block object.

    Parameters
    ----------
    new_payload_request :
        The new payload request (must be V5 format).

    Returns
    -------
    block : Block
        The block constructed from the payload.

    """
    execution_payload = new_payload_request[0]
    parent_beacon_block_root = new_payload_request[2]
    execution_requests = new_payload_request[3]

    # Convert withdrawals
    withdrawals: Tuple[Withdrawal, ...] = tuple(
        Withdrawal(
            index=U64(w.index),
            validator_index=U64(w.validator_index),
            address=Address(w.address),
            amount=U256(w.amount),
        )
        for w in (execution_payload.withdrawals or [])
    )

    # Compute withdrawals root
    withdrawals_trie: Trie[Bytes, Bytes] = Trie(secured=False, default=b"")
    for i, wd in enumerate(withdrawals):
        trie_set(withdrawals_trie, rlp.encode(Uint(i)), rlp.encode(wd))
    withdrawals_root = root(withdrawals_trie)

    # Prepare transactions
    transactions: Tuple[Bytes | LegacyTransaction, ...] = tuple(
        Bytes(tx) for tx in execution_payload.transactions
    )

    # Compute transactions root
    transactions_trie: Trie[Bytes, Bytes] = Trie(secured=False, default=b"")
    for i, tx in enumerate(transactions):
        trie_set(
            transactions_trie,
            rlp.encode(Uint(i)),
            tx if isinstance(tx, Bytes) else rlp.encode(tx),
        )
    transactions_root = root(transactions_trie)

    # Compute requests hash
    requests_hash = compute_requests_hash(list(execution_requests))

    header = Header(
        parent_hash=Hash32(execution_payload.parent_hash),
        ommers_hash=EMPTY_OMMER_HASH,
        coinbase=Address(execution_payload.fee_recipient),
        state_root=Root(execution_payload.state_root),
        transactions_root=transactions_root,
        receipt_root=Root(execution_payload.receipts_root),
        bloom=Bloom(execution_payload.logs_bloom),
        difficulty=Uint(0),
        number=Uint(execution_payload.number),
        gas_limit=Uint(execution_payload.gas_limit),
        gas_used=Uint(execution_payload.gas_used),
        timestamp=U256(execution_payload.timestamp),
        extra_data=Bytes(execution_payload.extra_data),
        prev_randao=Bytes32(execution_payload.prev_randao),
        nonce=Bytes8(b"\x00\x00\x00\x00\x00\x00\x00\x00"),
        base_fee_per_gas=Uint(execution_payload.base_fee_per_gas),
        withdrawals_root=withdrawals_root,
        blob_gas_used=U64(execution_payload.blob_gas_used or 0),
        excess_blob_gas=U64(execution_payload.excess_blob_gas or 0),
        parent_beacon_block_root=Root(parent_beacon_block_root),
        requests_hash=requests_hash,
    )

    return Block(
        header=header,
        transactions=transactions,
        ommers=(),
        withdrawals=withdrawals,
    )


def _build_node_map(nodes: List[Bytes]) -> Dict[Hash32, Bytes]:
    """
    Build a hash -> RLP node mapping from witness nodes.

    Parameters
    ----------
    nodes :
        List of RLP-encoded trie nodes from the witness.

    Returns
    -------
    node_map :
        Dictionary mapping node hash to RLP-encoded node.

    """
    node_map: Dict[Hash32, Bytes] = {}
    for node_rlp in nodes:
        node_hash = Hash32(keccak256(node_rlp))
        node_map[node_hash] = node_rlp
    return node_map


def _build_bytecode_map(bytecodes: List[Bytes]) -> Dict[Hash32, Bytes]:
    """
    Build a code_hash -> bytecode mapping from witness bytecodes.

    Parameters
    ----------
    bytecodes :
        List of bytecodes from the witness.

    Returns
    -------
    bytecode_map :
        Dictionary mapping code hash to bytecode.

    """
    bytecode_map: Dict[Hash32, Bytes] = {}
    for bytecode in bytecodes:
        code_hash = Hash32(keccak256(bytecode))
        bytecode_map[code_hash] = bytecode
    return bytecode_map


def _build_state_trie_from_witness(
    node_map: Dict[Hash32, Bytes],
    state_root: Root,
) -> WitnessBackedTrie:
    """
    Build the main state trie from witness nodes.

    This creates a WitnessBackedTrie that can be used for stateless execution.
    The trie stores raw RLP-encoded account data in its leaves; decoding
    to Account objects happens at access time.

    Parameters
    ----------
    node_map :
        Mapping from node hash to RLP-encoded node bytes.
    state_root :
        The state root from the parent block header.

    Returns
    -------
    state_trie :
        A witness-backed trie for the state.

    """
    return build_witness_trie(
        node_map=node_map,
        root_hash=state_root,
    )


def create_from_execution_witness(
    witness: ExecutionWitness,
    chain_id: Optional[U64] = None,
) -> WitnessBackedBlockChain:
    """
    Create a WitnessBackedBlockChain from an ExecutionWitness.

    This function reconstructs the minimal blockchain state needed to execute
    a block from the provided witness data. The returned blockchain uses
    witness-backed tries for state access instead of full state.

    Parameters
    ----------
    witness :
        The ExecutionWitness containing nodes, bytecodes, and ancestors.
    chain_id :
        The chain ID (defaults to 1 for mainnet if None).

    Returns
    -------
    blockchain : WitnessBackedBlockChain
        The blockchain object initialized from the witness.

    """
    if chain_id is None:
        chain_id = U64(1)

    # Step 1: Decode parent header to get state root
    parent_header = rlp.decode_to(Header, witness.ancestors[0])
    pre_state_root = parent_header.state_root

    # Step 2: Build node map from witness nodes
    node_map = _build_node_map(witness.nodes)

    # Step 3: Build bytecode map from witness bytecodes
    bytecode_map = _build_bytecode_map(witness.bytecodes)

    # Step 4: Build the pre-state trie from witness nodes
    state_trie = _build_state_trie_from_witness(node_map, pre_state_root)

    # Verify that the built trie has the expected root
    built_root = witness_trie_root(state_trie)
    if built_root != pre_state_root:
        raise InvalidBlock(
            f"Built trie root {built_root.hex()} does not match "
            f"expected state root {pre_state_root.hex()}"
        )

    # Step 5: Compute ancestor hashes
    ancestor_hashes = [
        Hash32(keccak256(ancestor_rlp)) for ancestor_rlp in witness.ancestors
    ]

    # Step 6: Create witness-backed state with two-layer architecture
    base_layer = WitnessBaseLayer(
        _main_trie=state_trie,
        _node_map=node_map,
        _bytecode_map=bytecode_map,
        _storage_tries_cache={},
    )
    diff_layer = DiffLayer(
        _accounts={},
        _storage={},
        _bytecodes={},
        _dirty_accounts=set(),
        _dirty_storage={},
        created_accounts=set(),
    )
    state = WitnessBackedState(
        _base=base_layer,
        _diff=diff_layer,
        _snapshots=[],
    )

    # Step 7: Return witness-backed blockchain
    return WitnessBackedBlockChain(
        _ancestors=list(witness.ancestors),
        _ancestor_hashes=ancestor_hashes,
        state=state,
        chain_id=chain_id,
    )


def tree_hash_root(new_payload_request: NewPayloadRequest) -> Hash32:
    """
    Compute the SSZ hash tree root of a NewPayloadRequest.

    Parameters
    ----------
    new_payload_request :
        The NewPayloadRequest object to compute the hash tree root for.

    Returns
    -------
    root : Hash32
        The SSZ hash tree root of the object.

    """
    # TODO: the repo doesn't import any SSZ library yet.
    # Implement this function when SSZ support is added.
    raise NotImplementedError("tree_hash_root requires SSZ support")


def stateless_state_transition(
    stateless_input: StatelessInput,
) -> StatelessOutput:
    """
    Execute a stateless state transition.

    This is the main entry point for stateless block validation. It validates
    the input, converts the payload to a block, and executes the state
    transition using only the provided witness data.

    Parameters
    ----------
    stateless_input :
        The stateless input containing the new payload request, witness, and
        public keys.

    Returns
    -------
    output : StatelessOutput
        The result of the stateless execution.

    """
    # Validate new_payload_request is EngineNewPayloadV5Parameters (4-tuple)
    new_payload_request = stateless_input.new_payload_request
    is_tuple = isinstance(new_payload_request, tuple)
    if not is_tuple or len(new_payload_request) != 4:
        return StatelessOutput(
            new_payload_request_root=Hash32(b"\x00" * 32),
            success=False,
        )

    if not validate_new_payload_params(new_payload_request):
        return StatelessOutput(
            new_payload_request_root=Hash32(b"\x00" * 32),
            success=False,
        )

    # Validate execution witness
    execution_payload = new_payload_request[0]
    parent_hash = Hash32(execution_payload.parent_hash)
    if not validate_execution_witness(stateless_input.witness, parent_hash):
        return StatelessOutput(
            new_payload_request_root=Hash32(b"\x00" * 32),
            success=False,
        )

    # Convert to Block
    block = block_from_new_payload_request(new_payload_request)

    # Create blockchain from witness and execute
    blockchain = create_from_execution_witness(stateless_input.witness)
    try:
        state_transition(blockchain, block)
        success = True
    except InvalidBlock:
        success = False
    new_payload_request_root = tree_hash_root(new_payload_request)
    return StatelessOutput(
        new_payload_request_root=new_payload_request_root,
        success=success,
    )
