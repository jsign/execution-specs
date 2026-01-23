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
    EMPTY_ACCOUNT,
    Account,
    Address,
    Bloom,
    Root,
    VersionedHash,
    encode_account,
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
    root,
    trie_set,
    witness_trie_get,
    witness_trie_root,
    witness_trie_set,
)


@dataclass
class WitnessBackedState:
    """
    State implementation backed by witness tries for stateless execution.

    Unlike the full State class which stores accounts in a regular Trie,
    this uses WitnessBackedTrie for lookups and modifications. Storage tries
    are built lazily from the witness node_map when first accessed.

    The trie stores raw RLP-encoded account data (nonce, balance,
    storage_root, code_hash). Bytecode is looked up from bytecode_map.
    """

    _main_trie: WitnessBackedTrie
    _storage_tries: Dict[Address, WitnessBackedTrie] = field(
        default_factory=dict
    )
    _node_map: Dict[Hash32, Bytes] = field(default_factory=dict)
    _bytecode_map: Dict[Hash32, Bytes] = field(default_factory=dict)
    _snapshots: List[
        Tuple[WitnessBackedTrie, Dict[Address, WitnessBackedTrie]]
    ] = field(default_factory=list)
    created_accounts: Set[Address] = field(default_factory=set)


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


def _decode_account_from_witness(
    encoded: Bytes,
    bytecode_map: Dict[Hash32, Bytes],
) -> Tuple[Account, Root]:
    """
    Decode an account from witness trie leaf value.

    The witness trie stores accounts as RLP(nonce, balance, storage_root,
    code_hash). The actual bytecode must be looked up from bytecode_map.

    Parameters
    ----------
    encoded :
        RLP-encoded account data from witness trie.
    bytecode_map :
        Mapping from code_hash to bytecode.

    Returns
    -------
    account : Account
        The decoded account with code populated from bytecode_map.
    storage_root : Root
        The storage root for building the storage trie.

    """
    decoded = rlp.decode(encoded)
    nonce = Uint(decoded[0])
    balance = U256(decoded[1])
    storage_root = Root(decoded[2])
    code_hash = Hash32(decoded[3])

    # Look up code from bytecode_map
    # Empty code has hash keccak256(b"") which may not be in bytecode_map
    empty_code_hash = Hash32(keccak256(b""))
    if code_hash == empty_code_hash:
        code = b""
    elif code_hash in bytecode_map:
        code = bytecode_map[code_hash]
    else:
        # Code hash not in witness - incomplete witness
        raise InvalidBlock(
            f"Code hash {code_hash.hex()} not in witness bytecode_map"
        )

    account = Account(nonce=nonce, balance=balance, code=code)
    return account, storage_root


def _get_storage_root_from_encoded_account(encoded: Bytes) -> Root:
    """Extract storage root from RLP-encoded account without full decode."""
    decoded = rlp.decode(encoded)
    return Root(decoded[2])


def witness_get_account_optional(
    state: WitnessBackedState,
    address: Address,
) -> Optional[Account]:
    """
    Get an account from witness-backed state.

    Returns None if the account doesn't exist (proven by witness structure).
    Raises InvalidBlock if witness is incomplete (hits StubNode).

    Parameters
    ----------
    state :
        The witness-backed state.
    address :
        Address to lookup.

    Returns
    -------
    account : Optional[Account]
        The account at address, or None if it doesn't exist.

    """
    encoded = witness_trie_get(state._main_trie, address)
    if encoded is None:
        return None

    account, _ = _decode_account_from_witness(encoded, state._bytecode_map)
    return account


def witness_get_account(
    state: WitnessBackedState, address: Address
) -> Account:
    """Get account from witness-backed state, or EMPTY_ACCOUNT if not found."""
    account = witness_get_account_optional(state, address)
    if account is None:
        return EMPTY_ACCOUNT
    return account


def witness_set_account(
    state: WitnessBackedState,
    address: Address,
    account: Optional[Account],
) -> None:
    """
    Set an account in witness-backed state.

    Parameters
    ----------
    state :
        The witness-backed state.
    address :
        Address to set.
    account :
        Account to set, or None to delete.

    """
    if account is None:
        # Delete account and its storage trie
        witness_trie_set(state._main_trie, address, b"")
        if address in state._storage_tries:
            del state._storage_tries[address]
    else:
        # Get current storage root (or empty if new account)
        current_encoded = witness_trie_get(state._main_trie, address)
        if current_encoded is not None:
            storage_root = _get_storage_root_from_encoded_account(
                current_encoded
            )
        else:
            storage_root = EMPTY_TRIE_ROOT

        # Encode and set
        encoded = encode_account(account, storage_root)
        witness_trie_set(state._main_trie, address, encoded)


def _get_or_build_storage_trie(
    state: WitnessBackedState,
    address: Address,
) -> WitnessBackedTrie:
    """
    Get or lazily build a storage trie for an account.

    If the storage trie hasn't been built yet, it's constructed from
    the witness node_map using the account's storage_root.

    Parameters
    ----------
    state :
        The witness-backed state.
    address :
        Address of the account.

    Returns
    -------
    storage_trie : WitnessBackedTrie
        The storage trie for the account.

    """
    if address in state._storage_tries:
        return state._storage_tries[address]

    # Get account's storage root
    encoded = witness_trie_get(state._main_trie, address)
    if encoded is None:
        # Account doesn't exist - create empty storage trie
        storage_trie = WitnessBackedTrie(root_node=None)
    else:
        storage_root = _get_storage_root_from_encoded_account(encoded)
        if storage_root == EMPTY_TRIE_ROOT:
            storage_trie = WitnessBackedTrie(root_node=None)
        else:
            storage_trie = build_witness_trie(state._node_map, storage_root)

    state._storage_tries[address] = storage_trie
    return storage_trie


def witness_get_storage(
    state: WitnessBackedState,
    address: Address,
    key: Bytes32,
) -> U256:
    """
    Get a storage value from witness-backed state.

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
    storage_trie = _get_or_build_storage_trie(state, address)
    encoded = witness_trie_get(storage_trie, key)
    if encoded is None:
        return U256(0)
    return U256(rlp.decode(encoded))


def witness_set_storage(
    state: WitnessBackedState,
    address: Address,
    key: Bytes32,
    value: U256,
) -> None:
    """
    Set a storage value in witness-backed state.

    Also updates the account's storage_root in the main trie.

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
    storage_trie = _get_or_build_storage_trie(state, address)

    if value == U256(0):
        witness_trie_set(storage_trie, key, b"")
    else:
        witness_trie_set(storage_trie, key, rlp.encode(value))

    # Update account's storage root in main trie
    new_storage_root = witness_trie_root(storage_trie)
    encoded = witness_trie_get(state._main_trie, address)
    assert encoded is not None, "Cannot set storage for non-existent account"
    account, _ = _decode_account_from_witness(encoded, state._bytecode_map)
    new_encoded = encode_account(account, new_storage_root)
    witness_trie_set(state._main_trie, address, new_encoded)


def witness_state_root(state: WitnessBackedState) -> Root:
    """
    Compute the state root from witness-backed state.

    Parameters
    ----------
    state :
        The witness-backed state.

    Returns
    -------
    root : Root
        The state root.

    """
    return witness_trie_root(state._main_trie)


def witness_storage_root(state: WitnessBackedState, address: Address) -> Root:
    """
    Compute the storage root for an account.

    Parameters
    ----------
    state :
        The witness-backed state.
    address :
        Address of the account.

    Returns
    -------
    root : Root
        The storage root.

    """
    if address in state._storage_tries:
        return witness_trie_root(state._storage_tries[address])
    return EMPTY_TRIE_ROOT



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

    # Step 6: Create witness-backed state
    state = WitnessBackedState(
        _main_trie=state_trie,
        _storage_tries={},
        _node_map=node_map,
        _bytecode_map=bytecode_map,
        _snapshots=[],
        created_accounts=set(),
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
