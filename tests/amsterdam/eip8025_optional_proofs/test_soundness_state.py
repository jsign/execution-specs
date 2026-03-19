"""Execution witness state soundness tests."""

import pytest
from execution_testing import (
    Account,
    Alloc,
    Block,
    BlockchainTestFiller,
    ExecutionWitnessStateExpectation,
    Op,
    Transaction,
)
from execution_testing.test_types.execution_witness.modifiers import (
    remove_state_node,
)

from .state_helpers import (
    as_storage,
    build_large_storage,
    collect_account_proof_nodes,
    collect_storage_delete_auxiliary_nodes,
    collect_storage_proof_nodes,
    large_storage_value,
)

pytestmark = pytest.mark.valid_from("Amsterdam")

REFERENCE_SPEC_GIT_PATH = "N/A"
REFERENCE_SPEC_VERSION = "N/A"


def test_soundness_state_missing_storage_proof_node(
    pre: Alloc,
    blockchain_test: BlockchainTestFiller,
) -> None:
    """Removing a required storage proof node should fail."""
    read_slot = 1
    write_slot = 2
    storage = build_large_storage([read_slot])
    proof_nodes = collect_storage_proof_nodes(storage, [read_slot])
    assert proof_nodes
    removed_node = sorted(proof_nodes)[0]

    contract = pre.deploy_contract(
        code=Op.SSTORE(write_slot, Op.SLOAD(read_slot)) + Op.STOP,
        storage=as_storage(storage),
    )
    sender = pre.fund_eoa()
    tx = Transaction(sender=sender, to=contract, gas_limit=500_000)
    post_storage = storage | {write_slot: storage[read_slot]}

    blockchain_test(
        pre=pre,
        blocks=[
            Block(
                txs=[tx],
                expected_execution_witness_state=(
                    ExecutionWitnessStateExpectation(
                        nodes_present=proof_nodes,
                    ).modify(remove_state_node(removed_node))
                ),
                expected_stateless_validation_success=False,
            )
        ],
        post={
            sender: Account(nonce=1),
            contract: Account(storage=post_storage),
        },
    )


def test_soundness_state_missing_absent_slot_proof_node(
    pre: Alloc,
    blockchain_test: BlockchainTestFiller,
) -> None:
    """Removing an absence-proof node for SSTORE should fail."""
    storage = build_large_storage([1, 2])
    insert_slot = 3
    insert_value = large_storage_value(insert_slot)
    proof_nodes = collect_storage_proof_nodes(storage, [insert_slot])
    assert proof_nodes
    removed_node = sorted(proof_nodes)[0]

    contract = pre.deploy_contract(
        code=Op.SSTORE(insert_slot, insert_value) + Op.STOP,
        storage=as_storage(storage),
    )
    sender = pre.fund_eoa()
    tx = Transaction(sender=sender, to=contract, gas_limit=500_000)
    post_storage = storage | {insert_slot: insert_value}

    blockchain_test(
        pre=pre,
        blocks=[
            Block(
                txs=[tx],
                expected_execution_witness_state=(
                    ExecutionWitnessStateExpectation(
                        nodes_present=proof_nodes,
                    ).modify(remove_state_node(removed_node))
                ),
                expected_stateless_validation_success=False,
            )
        ],
        post={
            sender: Account(nonce=1),
            contract: Account(storage=post_storage),
        },
    )


def test_soundness_state_missing_delete_auxiliary_node(
    pre: Alloc,
    blockchain_test: BlockchainTestFiller,
) -> None:
    """Removing a branch-collapse auxiliary node should fail."""
    delete_slot = 1
    storage = build_large_storage([delete_slot, 2])
    proof_nodes = collect_storage_proof_nodes(storage, [delete_slot])
    auxiliary_nodes = collect_storage_delete_auxiliary_nodes(
        storage, delete_slot
    )
    assert proof_nodes
    assert len(auxiliary_nodes) == 1
    removed_node = auxiliary_nodes[0]

    contract = pre.deploy_contract(
        code=Op.SSTORE(delete_slot, 0) + Op.STOP,
        storage=as_storage(storage),
    )
    sender = pre.fund_eoa()
    tx = Transaction(sender=sender, to=contract, gas_limit=500_000)

    blockchain_test(
        pre=pre,
        blocks=[
            Block(
                txs=[tx],
                expected_execution_witness_state=(
                    ExecutionWitnessStateExpectation(
                        nodes_present=proof_nodes + auxiliary_nodes,
                    ).modify(remove_state_node(removed_node))
                ),
                expected_stateless_validation_success=False,
            )
        ],
        post={
            sender: Account(nonce=1),
            contract: Account(storage={2: storage[2]}),
        },
    )


def test_soundness_state_missing_account_trie_proof_node(
    pre: Alloc,
    blockchain_test: BlockchainTestFiller,
) -> None:
    """Removing an account-trie proof node should fail a simple transfer."""
    sender = pre.fund_eoa()
    recipient = pre.fund_eoa(amount=1)
    proof_nodes = collect_account_proof_nodes(pre, [sender, recipient])
    assert proof_nodes
    removed_node = sorted(proof_nodes)[0]

    tx = Transaction(sender=sender, to=recipient, value=1, gas_limit=21_000)

    blockchain_test(
        pre=pre,
        blocks=[
            Block(
                txs=[tx],
                expected_execution_witness_state=(
                    ExecutionWitnessStateExpectation(
                        nodes_present=proof_nodes,
                    ).modify(remove_state_node(removed_node))
                ),
                expected_stateless_validation_success=False,
            )
        ],
        post={
            sender: Account(nonce=1),
            recipient: Account(balance=2),
        },
    )
