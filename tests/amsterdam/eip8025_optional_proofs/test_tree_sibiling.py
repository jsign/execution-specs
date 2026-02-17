"""
Witness-focused storage trie sibling test.

This case is designed to expose the Osaka deferred witness generation behavior
where storage inserts/updates are applied before deletions.
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


def test_storage_trie_insert_then_delete_sibling(
    pre: Alloc,
    blockchain_test: BlockchainTestFiller,
) -> None:
    """
    Pre-state has two storage slots (A and B).
    The transaction inserts slot C and deletes slot A.

    This is useful for inspecting execution witness minimization:
    when writes are applied insert/update-first, deleting A no longer
    collapses the A/B branch and should avoid forcing B into witness nodes.
    """
    sender = pre.fund_eoa()

    # Chosen so `keccak256(slot)` starts with different high nibbles:
    # slot 0x00 -> 0x2*, slot 0x01 -> 0xb*, slot 0x02 -> 0x4*.
    # This gives a pre-state storage root branch with two children (A, B),
    # and lets C become a third sibling before deleting A.
    slot_a = 0x00
    slot_b = 0x01
    slot_c = 0x02

    value_a = 0xAA
    value_b = 0xBB
    value_c = 0xCC

    contract = pre.deploy_contract(
        code=Op.SSTORE(slot_c, value_c) + Op.SSTORE(slot_a, 0) + Op.STOP,
        storage={
            slot_a: value_a,
            slot_b: value_b,
        },
    )

    tx = Transaction(
        sender=sender,
        to=contract,
        gas_limit=200_000,
    )

    post = {
        contract: Account(
            storage={
                slot_b: value_b,
                slot_c: value_c,
            }
        )
    }

    blockchain_test(
        pre=pre,
        blocks=[Block(txs=[tx])],
        post=post,
    )


def test_runtime_only_branch_compression_not_in_witness(
    pre: Alloc,
    blockchain_test: BlockchainTestFiller,
) -> None:
    """
    Runtime creates and then compresses a storage branch:
    1. Insert A
    2. Insert B (creates a branch)
    3. Delete A (compresses branch to surviving B path)

    Since storage is empty in pre-state, all storage trie nodes involved in
    this compression are created during block execution. They should therefore
    not be present in execution witness nodes (which only include pre-state
    nodes).
    """
    sender = pre.fund_eoa()

    # Distinct first nibbles after secure hashing, so A/B diverge at root.
    slot_a = 0x00
    slot_b = 0x01
    value_a = 0xAA
    value_b = 0xBB

    contract = pre.deploy_contract(
        # Creates a branch at runtime and then compresses it by deleting A.
        code=(
            Op.SSTORE(slot_a, value_a)
            + Op.SSTORE(slot_b, value_b)
            + Op.SSTORE(slot_a, 0)
            + Op.STOP
        ),
        storage={},
    )

    tx = Transaction(
        sender=sender,
        to=contract,
        gas_limit=250_000,
    )

    post = {
        contract: Account(
            storage={
                slot_b: value_b,
            }
        )
    }

    blockchain_test(
        pre=pre,
        blocks=[Block(txs=[tx])],
        post=post,
    )
