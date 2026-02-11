"""
State Trie.

.. contents:: Table of Contents
    :backlinks: none
    :local:

Introduction
------------

The state trie is the structure responsible for storing
`.fork_types.Account` objects.
"""

import copy
from dataclasses import dataclass, field
from typing import (
    Callable,
    Dict,
    Generic,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
    cast,
)

from ethereum_rlp import Extended, rlp
from ethereum_types.bytes import Bytes
from ethereum_types.frozen import slotted_freezable
from ethereum_types.numeric import U256, Uint
from typing_extensions import assert_type

from ethereum.crypto.hash import keccak256
from ethereum.forks.prague import trie as previous_trie
from ethereum.utils.hexadecimal import hex_to_bytes

from .blocks import Receipt, Withdrawal
from .fork_types import Account, Address, Root, encode_account
from .transactions import LegacyTransaction

# note: an empty trie (regardless of whether it is secured) has root:
#
#   keccak256(RLP(b''))
#       ==
#   56e81f171bcc55a6ff8345e692c0f86e5b48e01b996cadc001622fb5e363b421 # noqa: E501
#
# also:
#
#   keccak256(RLP(()))
#       ==
#   1dcc4de8dec75d7aab85b567b6ccd41ad312451b948a7413f0a142fd40d49347 # noqa: E501
#
# which is the sha3Uncles hash in block header with no uncles
EMPTY_TRIE_ROOT = Root(
    hex_to_bytes(
        "56e81f171bcc55a6ff8345e692c0f86e5b48e01b996cadc001622fb5e363b421"
    )
)

Node = (
    Account
    | Bytes
    | LegacyTransaction
    | Receipt
    | Uint
    | U256
    | Withdrawal
    | None
)
K = TypeVar("K", bound=Bytes)
V = TypeVar(
    "V",
    Optional[Account],
    Optional[Bytes],
    Bytes,
    Optional[LegacyTransaction | Bytes],
    Optional[Receipt | Bytes],
    Optional[Withdrawal | Bytes],
    Uint,
    U256,
)


@slotted_freezable
@dataclass
class LeafNode:
    """Leaf node in the Merkle Trie."""

    rest_of_key: Bytes
    value: Extended


@slotted_freezable
@dataclass
class ExtensionNode:
    """Extension node in the Merkle Trie."""

    key_segment: Bytes
    subnode: Extended


BranchSubnodes = Tuple[
    Extended,
    Extended,
    Extended,
    Extended,
    Extended,
    Extended,
    Extended,
    Extended,
    Extended,
    Extended,
    Extended,
    Extended,
    Extended,
    Extended,
    Extended,
    Extended,
]


@slotted_freezable
@dataclass
class BranchNode:
    """Branch node in the Merkle Trie."""

    subnodes: BranchSubnodes
    value: Extended


InternalNode = LeafNode | ExtensionNode | BranchNode


# Mutable node types for incremental MPT updates
@dataclass
class MutableLeafNode:
    """Mutable leaf node in the Merkle Trie for in-place updates."""

    rest_of_key: Bytes
    value: Bytes
    _hash: Optional[Bytes] = None  # Cached hash, invalidated on change
    _rlp: Optional[Bytes] = None  # Cached RLP encoding


@dataclass
class MutableExtensionNode:
    """Mutable extension node in the Merkle Trie for in-place updates."""

    key_segment: Bytes
    child: "MutableNode"
    _hash: Optional[Bytes] = None
    _rlp: Optional[Bytes] = None


@dataclass
class MutableBranchNode:
    """Mutable branch node in the Merkle Trie for in-place updates."""

    children: List[Optional["MutableNode"]]  # 16 children slots
    value: Bytes  # Value if key terminates at this branch
    _hash: Optional[Bytes] = None
    _rlp: Optional[Bytes] = None


MutableNode = Union[
    MutableLeafNode, MutableExtensionNode, MutableBranchNode, None
]


@dataclass
class Witness:
    """Tracks nodes accessed during trie operations for witness generation."""

    accessed_nodes: Dict[Bytes, Bytes] = field(
        default_factory=dict
    )  # hash -> RLP encoding
    accessed_keys: Set[Bytes] = field(default_factory=set)  # Original keys
    bytecodes: List[Bytes] = field(default_factory=list)  # Accessed bytecodes
    ancestors: List[Bytes] = field(default_factory=list)  # RLP-encoded headers


@dataclass
class IncrementalMPT(Generic[K, V]):
    """
    An MPT that supports incremental updates and witness tracking.

    This maintains an actual tree structure that can be updated in-place,
    rather than rebuilding the entire tree on each root calculation.
    """

    secured: bool
    default: V
    root_node: MutableNode = None
    witness: Witness = field(default_factory=Witness)
    _data: Dict[K, V] = field(default_factory=dict)  # For backward compat


def encode_internal_node(node: Optional[InternalNode]) -> Extended:
    """
    Encodes a Merkle Trie node into its RLP form. The RLP will then be
    serialized into a `Bytes` and hashed unless it is less than 32 bytes
    when serialized.

    This function also accepts `None`, representing the absence of a node,
    which is encoded to `b""`.

    Parameters
    ----------
    node : Optional[InternalNode]
        The node to encode.

    Returns
    -------
    encoded : `Extended`
        The node encoded as RLP.

    """
    unencoded: Extended
    if node is None:
        unencoded = b""
    elif isinstance(node, LeafNode):
        unencoded = (
            nibble_list_to_compact(node.rest_of_key, True),
            node.value,
        )
    elif isinstance(node, ExtensionNode):
        unencoded = (
            nibble_list_to_compact(node.key_segment, False),
            node.subnode,
        )
    elif isinstance(node, BranchNode):
        unencoded = list(node.subnodes) + [node.value]
    else:
        raise AssertionError(f"Invalid internal node type {type(node)}!")

    encoded = rlp.encode(unencoded)
    if len(encoded) < 32:
        return unencoded
    else:
        return keccak256(encoded)


def encode_node(node: Node, storage_root: Optional[Bytes] = None) -> Bytes:
    """
    Encode a Node for storage in the Merkle Trie.

    Currently mostly an unimplemented stub.
    """
    if isinstance(node, Account):
        assert storage_root is not None
        return encode_account(node, storage_root)
    elif isinstance(node, (LegacyTransaction, Receipt, Withdrawal, U256)):
        return rlp.encode(node)
    elif isinstance(node, Bytes):
        return node
    else:
        return previous_trie.encode_node(node, storage_root)


@dataclass
class Trie(Generic[K, V]):
    """
    The Merkle Trie.
    """

    secured: bool
    default: V
    _data: Dict[K, V] = field(default_factory=dict)


def copy_trie(trie: Trie[K, V]) -> Trie[K, V]:
    """
    Create a copy of `trie`. Since only frozen objects may be stored in tries,
    the contents are reused.

    Parameters
    ----------
    trie: `Trie`
        Trie to copy.

    Returns
    -------
    new_trie : `Trie[K, V]`
        A copy of the trie.

    """
    return Trie(trie.secured, trie.default, copy.copy(trie._data))


def trie_set(trie: Trie[K, V], key: K, value: V) -> None:
    """
    Stores an item in a Merkle Trie.

    This method deletes the key if `value == trie.default`, because the Merkle
    Trie represents the default value by omitting it from the trie.

    Parameters
    ----------
    trie: `Trie`
        Trie to store in.
    key : `Bytes`
        Key to lookup.
    value : `V`
        Node to insert at `key`.

    """
    if value == trie.default:
        if key in trie._data:
            del trie._data[key]
    else:
        trie._data[key] = value


def trie_get(trie: Trie[K, V], key: K) -> V:
    """
    Gets an item from the Merkle Trie.

    This method returns `trie.default` if the key is missing.

    Parameters
    ----------
    trie:
        Trie to lookup in.
    key :
        Key to lookup.

    Returns
    -------
    node : `V`
        Node at `key` in the trie.

    """
    return trie._data.get(key, trie.default)


def common_prefix_length(a: Sequence, b: Sequence) -> int:
    """
    Find the longest common prefix of two sequences.
    """
    for i in range(len(a)):
        if i >= len(b) or a[i] != b[i]:
            return i
    return len(a)


def nibble_list_to_compact(x: Bytes, is_leaf: bool) -> Bytes:
    """
    Compresses nibble-list into a standard byte array with a flag.

    A nibble-list is a list of byte values no greater than `15`. The flag is
    encoded in high nibble of the highest byte. The flag nibble can be broken
    down into two two-bit flags.

    Highest nibble::

        +---+---+----------+--------+
        | _ | _ | is_leaf | parity |
        +---+---+----------+--------+
          3   2      1         0


    The lowest bit of the nibble encodes the parity of the length of the
    remaining nibbles -- `0` when even and `1` when odd. The second lowest bit
    is used to distinguish leaf and extension nodes. The other two bits are not
    used.

    Parameters
    ----------
    x :
        Array of nibbles.
    is_leaf :
        True if this is part of a leaf node, or false if it is an extension
        node.

    Returns
    -------
    compressed : `bytearray`
        Compact byte array.

    """
    compact = bytearray()

    if len(x) % 2 == 0:  # ie even length
        compact.append(16 * (2 * is_leaf))
        for i in range(0, len(x), 2):
            compact.append(16 * x[i] + x[i + 1])
    else:
        compact.append(16 * ((2 * is_leaf) + 1) + x[0])
        for i in range(1, len(x), 2):
            compact.append(16 * x[i] + x[i + 1])

    return Bytes(compact)


def bytes_to_nibble_list(bytes_: Bytes) -> Bytes:
    """
    Converts a `Bytes` into to a sequence of nibbles (bytes with value < 16).

    Parameters
    ----------
    bytes_:
        The `Bytes` to convert.

    Returns
    -------
    nibble_list : `Bytes`
        The `Bytes` in nibble-list format.

    """
    nibble_list = bytearray(2 * len(bytes_))
    for byte_index, byte in enumerate(bytes_):
        nibble_list[byte_index * 2] = (byte & 0xF0) >> 4
        nibble_list[byte_index * 2 + 1] = byte & 0x0F
    return Bytes(nibble_list)


def _prepare_data(
    data: Mapping[K, V],
    secured: bool,
    get_storage_root: Optional[Callable[[Address], Root]] = None,
) -> Mapping[Bytes, Bytes]:
    """
    Prepares data for trie root calculation. Removes values that are empty,
    hashes the keys (if `secured == True`) and encodes all the nodes.

    Parameters
    ----------
    data :
        The key-value data to prepare.
    secured :
        Whether keys should be hashed.
    get_storage_root :
        Function to get the storage root of an account. Needed to encode
        `Account` objects.

    Returns
    -------
    out : `Mapping[ethereum.base_types.Bytes, Node]`
        Object with keys mapped to nibble-byte form.

    """
    mapped: MutableMapping[Bytes, Bytes] = {}

    for preimage, value in data.items():
        if isinstance(value, Account):
            assert get_storage_root is not None
            address = Address(preimage)
            encoded_value = encode_node(value, get_storage_root(address))
        else:
            encoded_value = encode_node(value)
        if encoded_value == b"":
            raise AssertionError
        key: Bytes
        if secured:
            # "secure" tries hash keys once before construction
            key = keccak256(preimage)
        else:
            key = preimage
        mapped[bytes_to_nibble_list(key)] = encoded_value

    return mapped


def _prepare_trie(
    trie: Trie[K, V],
    get_storage_root: Optional[Callable[[Address], Root]] = None,
) -> Mapping[Bytes, Bytes]:
    """
    Prepares the trie for root calculation. Removes values that are empty,
    hashes the keys (if `secured == True`) and encodes all the nodes.

    Parameters
    ----------
    trie :
        The `Trie` to prepare.
    get_storage_root :
        Function to get the storage root of an account. Needed to encode
        `Account` objects.

    Returns
    -------
    out : `Mapping[ethereum.base_types.Bytes, Node]`
        Object with keys mapped to nibble-byte form.

    """
    return _prepare_data(trie._data, trie.secured, get_storage_root)


def root(
    trie: Trie[K, V],
    get_storage_root: Optional[Callable[[Address], Root]] = None,
) -> Root:
    """
    Computes the root of a modified merkle patricia trie (MPT).

    Parameters
    ----------
    trie :
        `Trie` to get the root of.
    get_storage_root :
        Function to get the storage root of an account. Needed to encode
        `Account` objects.


    Returns
    -------
    root : `.fork_types.Root`
        MPT root of the underlying key-value pairs.

    """
    obj = _prepare_trie(trie, get_storage_root)

    root_node = encode_internal_node(patricialize(obj, Uint(0)))
    if len(rlp.encode(root_node)) < 32:
        return keccak256(rlp.encode(root_node))
    else:
        assert isinstance(root_node, Bytes)
        return Root(root_node)


def patricialize(
    obj: Mapping[Bytes, Bytes], level: Uint
) -> Optional[InternalNode]:
    """
    Structural composition function.

    Used to recursively patricialize and merkleize a dictionary. Includes
    memoization of the tree structure and hashes.

    Parameters
    ----------
    obj :
        Underlying trie key-value pairs, with keys in nibble-list format.
    level :
        Current trie level.

    Returns
    -------
    node : `ethereum.base_types.Bytes`
        Root node of `obj`.

    """
    if len(obj) == 0:
        return None

    arbitrary_key = next(iter(obj))

    # if leaf node
    if len(obj) == 1:
        leaf = LeafNode(arbitrary_key[level:], obj[arbitrary_key])
        return leaf

    # prepare for extension node check by finding max j such that all keys in
    # obj have the same key[i:j]
    substring = arbitrary_key[level:]
    prefix_length = len(substring)
    for key in obj:
        prefix_length = min(
            prefix_length, common_prefix_length(substring, key[level:])
        )

        # finished searching, found another key at the current level
        if prefix_length == 0:
            break

    # if extension node
    if prefix_length > 0:
        prefix = arbitrary_key[int(level) : int(level) + prefix_length]
        return ExtensionNode(
            prefix,
            encode_internal_node(
                patricialize(obj, level + Uint(prefix_length))
            ),
        )

    branches: List[MutableMapping[Bytes, Bytes]] = []
    for _ in range(16):
        branches.append({})
    value = b""
    for key in obj:
        if len(key) == level:
            # shouldn't ever have an account or receipt in an internal node
            if isinstance(obj[key], (Account, Receipt, Uint)):
                raise AssertionError
            value = obj[key]
        else:
            branches[key[level]][key] = obj[key]

    subnodes = tuple(
        encode_internal_node(patricialize(branches[k], level + Uint(1)))
        for k in range(16)
    )
    return BranchNode(
        cast(BranchSubnodes, assert_type(subnodes, Tuple[Extended, ...])),
        value,
    )


def _build_mutable_tree(
    obj: Mapping[Bytes, Bytes], level: Uint
) -> MutableNode:
    """
    Build a mutable tree structure from a prepared key-value mapping.

    This is similar to `patricialize()` but creates mutable nodes for
    in-place updates.

    Parameters
    ----------
    obj :
        Underlying trie key-value pairs, with keys in nibble-list format.
    level :
        Current trie level.

    Returns
    -------
    node : `MutableNode`
        Root node of the mutable tree.

    """
    if len(obj) == 0:
        return None

    arbitrary_key = next(iter(obj))

    # Leaf node case
    if len(obj) == 1:
        return MutableLeafNode(
            rest_of_key=arbitrary_key[level:],
            value=obj[arbitrary_key],
        )

    # Check for common prefix (extension node)
    substring = arbitrary_key[level:]
    prefix_length = len(substring)
    for key in obj:
        prefix_length = min(
            prefix_length, common_prefix_length(substring, key[level:])
        )
        if prefix_length == 0:
            break

    if prefix_length > 0:
        prefix = arbitrary_key[int(level) : int(level) + prefix_length]
        child = _build_mutable_tree(obj, level + Uint(prefix_length))
        return MutableExtensionNode(key_segment=prefix, child=child)

    # Branch node case
    branches: List[MutableMapping[Bytes, Bytes]] = [{} for _ in range(16)]
    value = b""

    for key in obj:
        if len(key) == level:
            value = obj[key]
        else:
            branches[key[level]][key] = obj[key]

    children: List[Optional[MutableNode]] = [
        _build_mutable_tree(branches[k], level + Uint(1)) for k in range(16)
    ]

    return MutableBranchNode(children=children, value=value)


def build_mpt(
    data: Mapping[K, V],
    secured: bool,
    default: V,
    get_storage_root: Optional[Callable[[Address], Root]] = None,
) -> IncrementalMPT[K, V]:
    """
    Build an IncrementalMPT from key-value data.

    This is called with the pre-execution state to create a mutable
    tree structure that can be updated in-place during execution.

    Parameters
    ----------
    data :
        The source key-value data to build from.
    secured :
        Whether to hash keys before insertion.
    default :
        Default value for missing keys.
    get_storage_root :
        Function to get the storage root of an account.

    Returns
    -------
    mpt : `IncrementalMPT[K, V]`
        An incremental MPT with the same data.

    """
    prepared = _prepare_data(data, secured, get_storage_root)
    root_node = _build_mutable_tree(prepared, Uint(0))

    return IncrementalMPT(
        secured=secured,
        default=default,
        root_node=root_node,
        _data=dict(data),
    )


def _invalidate_hash(node: MutableNode) -> None:
    """Invalidate the cached hash of a node."""
    if node is not None:
        node._hash = None
        node._rlp = None


def _record_witness(
    witness: Witness, node: MutableNode, key: Optional[Bytes] = None
) -> None:
    """Record a node access in the witness."""
    if node is None:
        return

    # Record the key if provided
    if key is not None:
        witness.accessed_keys.add(key)

    # Compute hash and RLP if not cached
    node_hash, node_rlp = _compute_node_hash_and_rlp(node)
    if node_hash is not None and node_hash not in witness.accessed_nodes:
        witness.accessed_nodes[node_hash] = node_rlp


def _encode_mutable_node(node: MutableNode) -> Extended:
    """
    Encode a mutable node to its RLP form (unencoded tuple or hash).

    Similar to encode_internal_node but for mutable nodes.
    """
    if node is None:
        return b""
    elif isinstance(node, MutableLeafNode):
        return (
            nibble_list_to_compact(node.rest_of_key, True),
            node.value,
        )
    elif isinstance(node, MutableExtensionNode):
        child_encoded = _encode_mutable_node_to_extended(node.child)
        return (
            nibble_list_to_compact(node.key_segment, False),
            child_encoded,
        )
    elif isinstance(node, MutableBranchNode):
        children_encoded = [
            _encode_mutable_node_to_extended(child) for child in node.children
        ]
        return children_encoded + [node.value]
    else:
        raise AssertionError(f"Invalid mutable node type {type(node)}!")


def _encode_mutable_node_to_extended(node: MutableNode) -> Extended:
    """
    Encode a mutable node for embedding in parent.

    Returns the hash if RLP >= 32 bytes, otherwise returns unencoded form.
    """
    if node is None:
        return b""

    unencoded = _encode_mutable_node(node)
    encoded = rlp.encode(unencoded)

    if len(encoded) < 32:
        return unencoded
    else:
        return keccak256(encoded)


def _compute_node_hash_and_rlp(
    node: MutableNode,
) -> Tuple[Optional[Bytes], Bytes]:
    """
    Compute the hash and RLP encoding of a node.

    Returns (hash, rlp) where hash may be None for small nodes.
    """
    if node is None:
        return None, b""

    # Use cached values if available
    if node._rlp is not None:
        if node._hash is not None:
            return node._hash, node._rlp
        elif len(node._rlp) >= 32:
            return keccak256(node._rlp), node._rlp

    unencoded = _encode_mutable_node(node)
    encoded = rlp.encode(unencoded)

    # Cache the RLP
    node._rlp = encoded

    if len(encoded) >= 32:
        node._hash = keccak256(encoded)
        return node._hash, encoded
    else:
        return None, encoded


def mpt_get(mpt: IncrementalMPT[K, V], key: K) -> V:
    """
    Get a value from the incremental MPT.

    Traverses the tree and records accessed nodes in the witness for
    execution witness generation.

    Parameters
    ----------
    mpt :
        The incremental MPT to get from.
    key :
        Key to lookup.

    Returns
    -------
    value : `V`
        Value at the key, or the default value if not found.

    """
    # Get from flat data for consistency
    value = mpt._data.get(key, mpt.default)

    # Record the key access
    if mpt.secured:
        nibble_key = bytes_to_nibble_list(keccak256(key))
    else:
        nibble_key = bytes_to_nibble_list(key)

    mpt.witness.accessed_keys.add(key)

    # Traverse tree and record witness nodes
    _mpt_traverse_for_witness(mpt, mpt.root_node, nibble_key, Uint(0))

    return value


def _mpt_traverse_for_witness(
    mpt: IncrementalMPT,
    node: MutableNode,
    key: Bytes,
    level: Uint,
) -> None:
    """Traverse the tree recording nodes in the witness."""
    if node is None:
        return

    _record_witness(mpt.witness, node)

    if isinstance(node, MutableLeafNode):
        # Leaf node - end of path
        pass
    elif isinstance(node, MutableExtensionNode):
        # Extension node - follow if key matches
        segment_len = len(node.key_segment)
        lvl = int(level)
        if key[lvl : lvl + segment_len] == node.key_segment:
            _mpt_traverse_for_witness(
                mpt, node.child, key, Uint(lvl + segment_len)
            )
    elif isinstance(node, MutableBranchNode):
        # Branch node - follow appropriate child
        lvl = int(level)
        if lvl < len(key):
            child_idx = key[lvl]
            _mpt_traverse_for_witness(
                mpt, node.children[child_idx], key, Uint(lvl + 1)
            )


def mpt_set(
    mpt: IncrementalMPT[K, V],
    key: K,
    value: V,
    get_storage_root: Optional[Callable[[Address], Root]] = None,
) -> None:
    """
    Set a value in the incremental MPT.

    Updates the tree in-place and invalidates cached hashes along the path.

    Parameters
    ----------
    mpt :
        The incremental MPT to update.
    key :
        Key to set.
    value :
        Value to set at the key.
    get_storage_root :
        Function to get storage root (for Account values).

    """
    # Update flat data for backward compatibility
    if value == mpt.default:
        if key in mpt._data:
            del mpt._data[key]
    else:
        mpt._data[key] = value

    # Prepare key and value
    if mpt.secured:
        nibble_key = bytes_to_nibble_list(keccak256(key))
    else:
        nibble_key = bytes_to_nibble_list(key)

    # Encode the value
    if value == mpt.default:
        encoded_value = b""
    elif isinstance(value, Account):
        assert get_storage_root is not None
        address = Address(key)
        encoded_value = encode_node(value, get_storage_root(address))
    else:
        encoded_value = encode_node(value)

    # Update tree
    if encoded_value == b"":
        # Delete operation
        mpt.root_node = _mpt_delete_node(
            mpt, mpt.root_node, nibble_key, Uint(0)
        )
    else:
        # Insert/update operation
        mpt.root_node = _mpt_insert_node(
            mpt, mpt.root_node, nibble_key, encoded_value, Uint(0)
        )


def _mpt_insert_node(
    mpt: IncrementalMPT,
    node: MutableNode,
    key: Bytes,
    value: Bytes,
    level: Uint,
) -> MutableNode:
    """
    Insert or update a value in the mutable tree.

    Returns the new/updated node for this position.
    """
    if node is None:
        # Empty slot - create new leaf
        return MutableLeafNode(rest_of_key=key[level:], value=value)

    _invalidate_hash(node)

    if isinstance(node, MutableLeafNode):
        return _insert_into_leaf(mpt, node, key, value, level)
    elif isinstance(node, MutableExtensionNode):
        return _insert_into_extension(mpt, node, key, value, level)
    elif isinstance(node, MutableBranchNode):
        return _insert_into_branch(mpt, node, key, value, level)
    else:
        raise AssertionError(f"Invalid node type {type(node)}")


def _insert_into_leaf(
    _mpt: IncrementalMPT,
    node: MutableLeafNode,
    key: Bytes,
    value: Bytes,
    level: Uint,
) -> MutableNode:
    """Handle insertion when current node is a leaf."""
    existing_key = node.rest_of_key
    remaining_key = key[level:]

    if existing_key == remaining_key:
        # Same key - update value
        node.value = value
        return node

    # Keys differ - need to create branch
    prefix_len = common_prefix_length(existing_key, remaining_key)

    # Create new branch or extension + branch
    if prefix_len > 0:
        # Common prefix - create extension then branch
        branch = _create_branch_from_two_leaves(
            existing_key[prefix_len:],
            node.value,
            remaining_key[prefix_len:],
            value,
        )
        return MutableExtensionNode(
            key_segment=existing_key[:prefix_len], child=branch
        )
    else:
        # No common prefix - create branch directly
        return _create_branch_from_two_leaves(
            existing_key, node.value, remaining_key, value
        )


def _create_branch_from_two_leaves(
    key1: Bytes, value1: Bytes, key2: Bytes, value2: Bytes
) -> MutableBranchNode:
    """Create a branch node from two key-value pairs."""
    children: List[Optional[MutableNode]] = [None] * 16
    branch_value = b""

    if len(key1) == 0:
        branch_value = value1
    else:
        idx1 = key1[0]
        children[idx1] = MutableLeafNode(rest_of_key=key1[1:], value=value1)

    if len(key2) == 0:
        branch_value = value2
    else:
        idx2 = key2[0]
        children[idx2] = MutableLeafNode(rest_of_key=key2[1:], value=value2)

    return MutableBranchNode(children=children, value=branch_value)


def _insert_into_extension(
    mpt: IncrementalMPT,
    node: MutableExtensionNode,
    key: Bytes,
    value: Bytes,
    level: Uint,
) -> MutableNode:
    """Handle insertion when current node is an extension."""
    remaining_key = key[level:]
    segment = node.key_segment
    prefix_len = common_prefix_length(segment, remaining_key)

    if prefix_len == len(segment):
        # Key follows extension completely - recurse into child
        node.child = _mpt_insert_node(
            mpt, node.child, key, value, level + Uint(prefix_len)
        )
        return node

    # Extension needs to be split
    if prefix_len > 0:
        # Partial match - create new extension for common prefix
        new_child = _split_extension(node, remaining_key, value, prefix_len)
        return MutableExtensionNode(
            key_segment=segment[:prefix_len], child=new_child
        )
    else:
        # No common prefix - create branch at this level
        return _split_extension(node, remaining_key, value, 0)


def _split_extension(
    node: MutableExtensionNode,
    remaining_key: Bytes,
    value: Bytes,
    prefix_len: int,
) -> MutableNode:
    """Split an extension node when keys diverge."""
    segment = node.key_segment
    children: List[Optional[MutableNode]] = [None] * 16
    branch_value = b""

    # Place existing extension's child
    segment_after_prefix = segment[prefix_len:]
    if len(segment_after_prefix) == 1:
        # Single nibble left - place child directly in branch
        idx = segment_after_prefix[0]
        children[idx] = node.child
    elif len(segment_after_prefix) > 1:
        # Multiple nibbles - create new extension
        idx = segment_after_prefix[0]
        children[idx] = MutableExtensionNode(
            key_segment=segment_after_prefix[1:], child=node.child
        )

    # Place new value
    key_after_prefix = remaining_key[prefix_len:]
    if len(key_after_prefix) == 0:
        branch_value = value
    else:
        idx = key_after_prefix[0]
        if children[idx] is None:
            children[idx] = MutableLeafNode(
                rest_of_key=key_after_prefix[1:], value=value
            )
        else:
            # Need to merge with existing child (shouldn't happen normally)
            raise AssertionError("Unexpected collision during split")

    return MutableBranchNode(children=children, value=branch_value)


def _insert_into_branch(
    mpt: IncrementalMPT,
    node: MutableBranchNode,
    key: Bytes,
    value: Bytes,
    level: Uint,
) -> MutableNode:
    """Handle insertion when current node is a branch."""
    remaining_key = key[level:]

    if len(remaining_key) == 0:
        # Value terminates at this branch
        node.value = value
        return node

    # Recurse into appropriate child
    child_idx = remaining_key[0]
    node.children[child_idx] = _mpt_insert_node(
        mpt, node.children[child_idx], key, value, level + Uint(1)
    )
    return node


def _mpt_delete_node(
    mpt: IncrementalMPT,
    node: MutableNode,
    key: Bytes,
    level: Uint,
) -> MutableNode:
    """
    Delete a key from the mutable tree.

    Returns the updated node (may be different type or None).
    """
    if node is None:
        return None

    _invalidate_hash(node)

    if isinstance(node, MutableLeafNode):
        if node.rest_of_key == key[level:]:
            return None  # Key found, delete
        return node  # Key not found, no change
    elif isinstance(node, MutableExtensionNode):
        return _delete_from_extension(mpt, node, key, level)
    elif isinstance(node, MutableBranchNode):
        return _delete_from_branch(mpt, node, key, level)
    else:
        raise AssertionError(f"Invalid node type {type(node)}")


def _delete_from_extension(
    mpt: IncrementalMPT,
    node: MutableExtensionNode,
    key: Bytes,
    level: Uint,
) -> MutableNode:
    """Handle deletion when current node is an extension."""
    segment = node.key_segment
    remaining_key = key[level:]
    prefix_len = common_prefix_length(segment, remaining_key)

    if prefix_len < len(segment):
        return node  # Key doesn't follow this extension

    # Recurse into child
    new_child = _mpt_delete_node(
        mpt, node.child, key, level + Uint(len(segment))
    )

    if new_child is None:
        return None

    # Collapse if child is now an extension
    if isinstance(new_child, MutableExtensionNode):
        return MutableExtensionNode(
            key_segment=segment + new_child.key_segment,
            child=new_child.child,
        )
    elif isinstance(new_child, MutableLeafNode):
        # Merge extension into leaf
        return MutableLeafNode(
            rest_of_key=segment + new_child.rest_of_key,
            value=new_child.value,
        )

    node.child = new_child
    return node


def _delete_from_branch(
    mpt: IncrementalMPT,
    node: MutableBranchNode,
    key: Bytes,
    level: Uint,
) -> MutableNode:
    """Handle deletion when current node is a branch."""
    remaining_key = key[level:]

    if len(remaining_key) == 0:
        # Delete value at this branch
        node.value = b""
    else:
        # Delete from child
        child_idx = remaining_key[0]
        node.children[child_idx] = _mpt_delete_node(
            mpt, node.children[child_idx], key, level + Uint(1)
        )

    # Check if branch can be collapsed
    return _collapse_branch(mpt, node)


def _collapse_branch(
    mpt: IncrementalMPT, node: MutableBranchNode
) -> MutableNode:
    """Collapse a branch node if it has only one child and no value."""
    non_empty = [(i, c) for i, c in enumerate(node.children) if c is not None]

    if len(non_empty) == 0 and node.value == b"":
        return None

    if len(non_empty) == 1 and node.value == b"":
        idx, child = non_empty[0]
        _record_witness(mpt.witness, child)  # Record the surviving child
        nibble = Bytes([idx])

        if isinstance(child, MutableLeafNode):
            return MutableLeafNode(
                rest_of_key=nibble + child.rest_of_key,
                value=child.value,
            )
        elif isinstance(child, MutableExtensionNode):
            return MutableExtensionNode(
                key_segment=nibble + child.key_segment,
                child=child.child,
            )
        else:
            # Child is a branch - create extension
            return MutableExtensionNode(key_segment=nibble, child=child)

    if len(non_empty) == 0 and node.value != b"":
        # Only value at this branch - convert to leaf
        return MutableLeafNode(rest_of_key=b"", value=node.value)

    return node


def mpt_root(mpt: IncrementalMPT) -> Root:
    """
    Compute the root hash of the incremental MPT.

    Uses cached hashes where available for efficiency.

    Parameters
    ----------
    mpt :
        The incremental MPT.

    Returns
    -------
    root : `Root`
        The MPT root hash.

    """
    if mpt.root_node is None:
        return EMPTY_TRIE_ROOT

    root_encoded = _encode_mutable_node_to_extended(mpt.root_node)

    if isinstance(root_encoded, Bytes):
        return Root(root_encoded)
    else:
        return keccak256(rlp.encode(root_encoded))
