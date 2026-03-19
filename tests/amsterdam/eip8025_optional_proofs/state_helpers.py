"""Helpers for `ExecutionWitness.state` tests."""

from collections.abc import Mapping, Sequence

from ethereum_types.bytes import Bytes as EthereumBytes
from ethereum_types.bytes import Bytes32
from ethereum_types.numeric import U256, Uint
from execution_testing import Alloc, Bytes, Storage
from execution_testing.forks import Amsterdam

from ethereum.forks.amsterdam.incremental_mpt import (
    build_mpt,
    mpt_get,
    mpt_set,
)
from ethereum.forks.amsterdam.state import (
    State as AmsterdamState,
    set_account,
    set_storage,
    store_code,
)
from ethereum.forks.amsterdam.trie import EMPTY_TRIE_ROOT, root
from ethereum.state import Account as EthereumAccount
from ethereum.state import Address as EthereumAddress


def large_storage_value(slot: int) -> int:
    """Return a 32-byte non-zero value so the leaf node is hashed."""
    return (1 << 255) + slot


def build_large_storage(slots: Sequence[int]) -> dict[int, int]:
    """Build a slot->value mapping with large deterministic values."""
    return {slot: large_storage_value(slot) for slot in slots}


def as_storage(storage: Mapping[int, int]) -> Storage:
    """Convert an int-keyed storage mapping into a `Storage`."""
    return Storage.model_validate(dict(storage.items()))


def _slot(slot: int) -> Bytes32:
    """Convert an integer slot into a 32-byte storage key."""
    return Bytes32(slot.to_bytes(32, byteorder="big"))


def _storage_mpt_input(storage: Mapping[int, int]) -> dict[Bytes32, U256]:
    """Convert int-based storage into the internal MPT key/value types."""
    return {_slot(slot): U256(value) for slot, value in storage.items()}


def _collect_storage_node_set(
    storage: Mapping[int, int],
    slots: Sequence[int],
) -> set[bytes]:
    """Collect hashed witness nodes for the given storage proof paths."""
    storage_mpt = build_mpt(
        _storage_mpt_input(storage), secured=True, default=U256(0)
    )
    for slot in slots:
        mpt_get(storage_mpt, _slot(slot))
    return set(storage_mpt.witness.accessed_nodes.values())


def _nodes(nodes: set[bytes]) -> list[Bytes]:
    """Return nodes as execution-testing bytes."""
    return [Bytes(node) for node in nodes]


def collect_storage_proof_nodes(
    storage: Mapping[int, int],
    slots: Sequence[int],
) -> list[Bytes]:
    """Collect the pre-state proof nodes for the given storage slots."""
    return _nodes(_collect_storage_node_set(storage, slots))


def collect_storage_delete_auxiliary_nodes(
    storage: Mapping[int, int],
    slot_to_delete: int,
) -> list[Bytes]:
    """Collect nodes added only because a delete compressed the trie."""
    storage_mpt = build_mpt(
        _storage_mpt_input(storage), secured=True, default=U256(0)
    )
    mpt_get(storage_mpt, _slot(slot_to_delete))
    before = set(storage_mpt.witness.accessed_nodes.values())
    mpt_set(storage_mpt, _slot(slot_to_delete), U256(0))
    after = set(storage_mpt.witness.accessed_nodes.values())
    return _nodes(after - before)


def collect_storage_path_only_nodes(
    storage: Mapping[int, int],
    slot: int,
    relative_to_slots: Sequence[int],
) -> list[Bytes]:
    """Collect nodes unique to one proof path relative to others."""
    slot_nodes = _collect_storage_node_set(storage, [slot])
    reference_nodes = _collect_storage_node_set(storage, relative_to_slots)
    return _nodes(slot_nodes - reference_nodes)


def collect_storage_post_state_only_nodes(
    pre_storage: Mapping[int, int],
    post_storage: Mapping[int, int],
    slot: int,
    pre_state_reference_slots: Sequence[int],
) -> list[Bytes]:
    """Collect nodes that appear only on the post-state proof path."""
    post_state_nodes = _collect_storage_node_set(post_storage, [slot])
    pre_state_nodes = _collect_storage_node_set(
        pre_storage, pre_state_reference_slots
    )
    return _nodes(post_state_nodes - pre_state_nodes)


def _build_pre_state(alloc: Alloc) -> AmsterdamState:
    """Build an Amsterdam state from an execution-testing alloc."""
    effective_alloc = Alloc.merge(
        Alloc.model_validate(Amsterdam.pre_allocation_blockchain()),
        alloc,
    )
    state = AmsterdamState()
    for address, account in effective_alloc.root.items():
        if account is None:
            continue

        ethereum_address = EthereumAddress(bytes(address))
        code_hash = store_code(
            state, EthereumBytes(bytes(account.code))
        )
        set_account(
            state,
            ethereum_address,
            EthereumAccount(
                nonce=Uint(int(account.nonce)),
                balance=U256(int(account.balance)),
                code_hash=code_hash,
            ),
        )
        for key, value in account.storage.root.items():
            set_storage(
                state,
                ethereum_address,
                Bytes32(int(key).to_bytes(32, byteorder="big")),
                U256(int(value)),
            )
    return state


def collect_account_proof_nodes(
    alloc: Alloc,
    addresses: Sequence[bytes],
) -> list[Bytes]:
    """Collect the pre-state account-trie proof nodes for the given addresses."""
    state = _build_pre_state(alloc)

    def get_storage_root(address: EthereumAddress) -> bytes:
        trie = state._storage_tries.get(address)
        if trie is None:
            return EMPTY_TRIE_ROOT
        return root(trie)

    account_mpt = build_mpt(
        state._main_trie._data,
        secured=True,
        default=None,
        get_storage_root=get_storage_root,
    )
    for address in addresses:
        mpt_get(account_mpt, EthereumAddress(bytes(address)))

    return _nodes(set(account_mpt.witness.accessed_nodes.values()))
