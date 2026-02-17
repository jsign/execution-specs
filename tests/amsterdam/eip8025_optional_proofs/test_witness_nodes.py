"""
Witness-focused node coverage for EIP-8025 optional proofs.

These tests are written to produce witness vectors that stress:
- storage branch compression ordering,
- dirty storage paths with no explicit reads,
- runtime-only storage mutations.
"""

import pytest
from execution_testing import (
    Account,
    Alloc,
    Block,
    BlockchainTestFiller,
    Op,
    Transaction,
)

pytestmark = pytest.mark.valid_at("Osaka")

REFERENCE_SPEC_GIT_PATH = "TODO"
REFERENCE_SPEC_VERSION = "TODO"


def test_witness_branch_compression_delete_then_insert_sibling(
    pre: Alloc,
    blockchain_test: BlockchainTestFiller,
) -> None:
    """
    Pre-state has sibling slots A and B. Runtime deletes A, then inserts C.

    This complements the existing insert-then-delete coverage by exercising
    the opposite write order over the same branch shape.
    """
    sender = pre.fund_eoa()

    slot_a = 0x00
    slot_b = 0x01
    slot_c = 0x02

    value_a = 0xAA
    value_b = 0xBB
    value_c = 0xCC

    contract = pre.deploy_contract(
        code=Op.SSTORE(slot_a, 0) + Op.SSTORE(slot_c, value_c) + Op.STOP,
        storage={
            slot_a: value_a,
            slot_b: value_b,
        },
    )

    tx = Transaction(sender=sender, to=contract, gas_limit=200_000)

    blockchain_test(
        pre=pre,
        blocks=[Block(txs=[tx])],
        post={
            contract: Account(
                storage={
                    slot_b: value_b,
                    slot_c: value_c,
                }
            )
        },
    )


def test_witness_branch_compression_update_then_delete_multiple_siblings(
    pre: Alloc,
    blockchain_test: BlockchainTestFiller,
) -> None:
    """
    Pre-state has three sibling slots A/B/C.
    Runtime updates C, then deletes A and B.

    This stresses branch compression when multiple leaves are removed while
    another pre-state leaf survives with an updated value.
    """
    sender = pre.fund_eoa()

    slot_a = 0x00
    slot_b = 0x01
    slot_c = 0x02

    value_a = 0xAA
    value_b = 0xBB
    value_c = 0xCC
    new_value_c = 0xDD

    contract = pre.deploy_contract(
        code=(
            Op.SSTORE(slot_c, new_value_c)
            + Op.SSTORE(slot_a, 0)
            + Op.SSTORE(slot_b, 0)
            + Op.STOP
        ),
        storage={
            slot_a: value_a,
            slot_b: value_b,
            slot_c: value_c,
        },
    )

    tx = Transaction(sender=sender, to=contract, gas_limit=250_000)

    blockchain_test(
        pre=pre,
        blocks=[Block(txs=[tx])],
        post={
            contract: Account(
                storage={
                    slot_c: new_value_c,
                }
            )
        },
    )


def test_witness_nodes_dirty_slots_without_reads(
    pre: Alloc,
    blockchain_test: BlockchainTestFiller,
) -> None:
    """
    Touch only dirty storage paths (no SLOADs): update one key, delete one key.

    This is a regression target for the deferred witness build path where dirty
    keys must still traverse pre-state storage trie nodes.
    """
    sender = pre.fund_eoa()

    slot_update = 0x11
    slot_delete = 0x22

    contract = pre.deploy_contract(
        code=(
            Op.SSTORE(slot_update, 0x1234)
            + Op.SSTORE(slot_delete, 0)
            + Op.STOP
        ),
        storage={
            slot_update: 0x01,
            slot_delete: 0x02,
        },
    )

    tx = Transaction(sender=sender, to=contract, gas_limit=200_000)

    blockchain_test(
        pre=pre,
        blocks=[Block(txs=[tx])],
        post={
            contract: Account(
                storage={
                    slot_update: 0x1234,
                }
            )
        },
    )


def test_witness_nodes_delete_all_prestate_slots(
    pre: Alloc,
    blockchain_test: BlockchainTestFiller,
) -> None:
    """
    Delete every pre-state storage key from a contract.

    The resulting storage trie is empty, while witness nodes must still reflect
    the pre-state paths touched by deletions.
    """
    sender = pre.fund_eoa()

    slot_a = 0x30
    slot_b = 0x31

    contract = pre.deploy_contract(
        code=Op.SSTORE(slot_a, 0) + Op.SSTORE(slot_b, 0) + Op.STOP,
        storage={
            slot_a: 0xAA,
            slot_b: 0xBB,
        },
    )

    tx = Transaction(sender=sender, to=contract, gas_limit=180_000)

    blockchain_test(
        pre=pre,
        blocks=[Block(txs=[tx])],
        post={contract: Account(storage={})},
    )


def test_witness_nodes_runtime_only_create_and_delete_to_empty(
    pre: Alloc,
    blockchain_test: BlockchainTestFiller,
) -> None:
    """
    Create and then delete storage keys in a contract that starts with empty
    storage.

    All affected storage trie nodes are runtime-only and should not be part of
    pre-state witness node capture.
    """
    sender = pre.fund_eoa()

    slot_a = 0x40
    slot_b = 0x41

    contract = pre.deploy_contract(
        code=(
            Op.SSTORE(slot_a, 0xAB)
            + Op.SSTORE(slot_b, 0xCD)
            + Op.SSTORE(slot_a, 0)
            + Op.SSTORE(slot_b, 0)
            + Op.STOP
        ),
        storage={},
    )

    tx = Transaction(sender=sender, to=contract, gas_limit=250_000)

    blockchain_test(
        pre=pre,
        blocks=[Block(txs=[tx])],
        post={contract: Account(storage={})},
    )


def test_witness_nodes_missing_slot_read_only(
    pre: Alloc,
    blockchain_test: BlockchainTestFiller,
) -> None:
    """
    Read a missing storage key from a contract with non-empty pre-state.

    This targets proof-of-absence traversal for `SLOAD` without any writes.
    """
    sender = pre.fund_eoa()

    slot_existing = 0x50
    slot_missing = 0x51
    existing_value = 0xAA

    contract = pre.deploy_contract(
        code=Op.POP(Op.SLOAD(slot_missing)) + Op.STOP,
        storage={slot_existing: existing_value},
    )

    tx = Transaction(sender=sender, to=contract, gas_limit=180_000)

    blockchain_test(
        pre=pre,
        blocks=[Block(txs=[tx])],
        post={contract: Account(storage={slot_existing: existing_value})},
    )


def test_witness_nodes_transient_new_slot_on_non_empty_prestate(
    pre: Alloc,
    blockchain_test: BlockchainTestFiller,
) -> None:
    """
    Insert then delete a new key while another pre-state key survives.

    This isolates runtime-only branch activity under a non-empty pre-state
    storage root.
    """
    sender = pre.fund_eoa()

    slot_anchor = 0x60
    slot_transient = 0x61
    anchor_value = 0xAA

    contract = pre.deploy_contract(
        code=(
            Op.SSTORE(slot_transient, 0x1234)
            + Op.SSTORE(slot_transient, 0)
            + Op.STOP
        ),
        storage={slot_anchor: anchor_value},
    )

    tx = Transaction(sender=sender, to=contract, gas_limit=200_000)

    blockchain_test(
        pre=pre,
        blocks=[Block(txs=[tx])],
        post={contract: Account(storage={slot_anchor: anchor_value})},
    )
