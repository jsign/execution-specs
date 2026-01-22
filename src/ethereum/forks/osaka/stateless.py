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

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Union

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
from .fork import (
    EMPTY_OMMER_HASH,
    state_transition,
)
from .fork_types import Address, Bloom, Root, VersionedHash
from .requests import compute_requests_hash
from .transactions import BlobTransaction, LegacyTransaction, decode_transaction
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


def validate_new_payload_params(new_payload_request: NewPayloadRequest) -> bool:
    """
    Validate blob versioned hashes and execution requests per Engine API spec.

    From engine_newPayloadV3 (Cancun):
    - Verify expected blob versioned hashes match actual hashes from transactions

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
    # Validate ancestors chain (checks non-empty, hash matches, chain integrity)
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


def create_from_execution_witness(witness: ExecutionWitness) -> "BlockChain":
    """
    Create a BlockChain object from an ExecutionWitness.

    This function reconstructs the minimal blockchain state needed to execute
    a block from the provided witness data.

    Parameters
    ----------
    witness :
        The ExecutionWitness containing nodes, bytecodes, and ancestors.

    Returns
    -------
    blockchain : BlockChain
        The blockchain object initialized from the witness.

    """
    # Step 1: Decode parent header to get state root
    parent_header = rlp.decode_to(Header, witness.ancestors[0])
    state_root = parent_header.state_root

    # Step 2: Build node map from witness nodes
    node_map = _build_node_map(witness.nodes)

    # Step 3: Build bytecode map from witness bytecodes
    bytecode_map = _build_bytecode_map(witness.bytecodes)

    # Step 4: Build the pre-state trie from witness nodes
    # This creates a WitnessBackedTrie that can be used for lookups and
    # modifications during stateless execution.
    state_trie = _build_state_trie_from_witness(node_map, state_root)

    # Verify that the built trie has the expected root
    built_root = witness_trie_root(state_trie)
    if built_root != state_root:
        raise InvalidBlock(
            f"Built trie root {built_root.hex()} does not match "
            f"expected state root {state_root.hex()}"
        )

    # TODO: Points 2 and 3 are not yet implemented:
    # 2. Creating a State object with the reconstructed trie
    # 3. Building a BlockChain object with the state and ancestor headers
    #
    # For now, we store the built trie components for future use:
    # - state_trie: WitnessBackedTrie for account lookups
    # - node_map: For building storage tries when needed
    # - bytecode_map: For looking up contract code by hash
    _ = state_trie  # Built state trie (point 1 complete)
    _ = bytecode_map  # For code lookups
    _ = node_map  # For storage trie building

    raise NotImplementedError(
        "create_from_execution_witness: points 2-3 not yet implemented"
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
    if not isinstance(new_payload_request, tuple) or len(new_payload_request) != 4:
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
