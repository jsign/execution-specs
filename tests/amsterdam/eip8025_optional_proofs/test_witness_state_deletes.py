"""Witness state collection scenarios for storage and account deletes."""

import pytest
from execution_testing import (
    Account,
    Alloc,
    Block,
    BlockchainTestFiller,
    ExecutionWitnessStateExpectation,
    Op,
    Transaction,
    compute_create2_address,
)

from .spec import ref_spec_8025
from .state_helpers import (
    as_storage,
    build_large_storage,
    collect_account_delete_auxiliary_nodes,
    collect_account_proof_nodes,
    collect_storage_delete_auxiliary_nodes,
    collect_storage_path_only_nodes,
    collect_storage_post_state_only_nodes,
    collect_storage_proof_nodes,
    find_account_ending_absence_proof,
    large_storage_value,
    merge_with_amsterdam_pre_alloc,
)

pytestmark = pytest.mark.valid_from("Amsterdam")

REFERENCE_SPEC_GIT_PATH = ref_spec_8025.git_path
REFERENCE_SPEC_VERSION = ref_spec_8025.version


def test_witness_state_sstore_delete_branch_collapse_adds_auxiliary_node(
    pre: Alloc,
    blockchain_test: BlockchainTestFiller,
) -> None:
    """
    Deleting slot 1 from the `{1, 2}` trie shape forces branch collapse.

    The witness should contain both the normal proof for slot 1 and the
    auxiliary node needed to preserve the untouched sibling subtree.
    """
    storage = build_large_storage([1, 2])
    proof_nodes = collect_storage_proof_nodes(storage, [1])
    auxiliary_nodes = collect_storage_delete_auxiliary_nodes(storage, 1)
    assert proof_nodes
    # Double check the auxiliary node after the deletion resulted in
    # exactly one MPT node.
    assert len(auxiliary_nodes) == 1

    contract = pre.deploy_contract(
        code=Op.SSTORE(1, 0) + Op.STOP,
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
                    )
                ),
            )
        ],
        post={
            sender: Account(nonce=1),
            contract: Account(storage={2: storage[2]}),
        },
    )


def test_witness_state_sstore_delete_without_collapse_omits_sibling_nodes(
    pre: Alloc,
    blockchain_test: BlockchainTestFiller,
) -> None:
    """
    Deleting slot 1 from the `{1, 2, 3}` trie shape does not collapse.

    The untouched sibling paths should remain absent from the witness.
    """
    storage = build_large_storage([1, 2, 3])
    proof_nodes = collect_storage_proof_nodes(storage, [1])
    auxiliary_nodes = collect_storage_delete_auxiliary_nodes(storage, 1)
    slot_2_only_nodes = collect_storage_path_only_nodes(storage, 2, [1])
    slot_3_only_nodes = collect_storage_path_only_nodes(storage, 3, [1])
    assert proof_nodes
    assert not auxiliary_nodes
    assert slot_2_only_nodes
    assert slot_3_only_nodes

    contract = pre.deploy_contract(
        code=Op.SSTORE(1, 0) + Op.STOP,
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
                        nodes_present=proof_nodes,
                        nodes_absent=slot_2_only_nodes + slot_3_only_nodes,
                    )
                ),
            )
        ],
        post={
            sender: Account(nonce=1),
            contract: Account(storage={2: storage[2], 3: storage[3]}),
        },
    )


def test_witness_state_sstore_delete_only_slot_keeps_proof(
    pre: Alloc,
    blockchain_test: BlockchainTestFiller,
) -> None:
    """
    Deleting the only slot should keep its pre-state proof and empty storage.
    """
    storage = build_large_storage([1])
    proof_nodes = collect_storage_proof_nodes(storage, [1])
    auxiliary_nodes = collect_storage_delete_auxiliary_nodes(storage, 1)
    assert proof_nodes
    assert not auxiliary_nodes

    contract = pre.deploy_contract(
        code=Op.SSTORE(1, 0) + Op.STOP,
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
                        nodes_present=proof_nodes,
                    )
                ),
            )
        ],
        post={
            sender: Account(nonce=1),
            contract: Account(storage={}),
        },
    )


@pytest.mark.parametrize(
    "pre_slots,new_slot_2_value",
    [
        pytest.param([1], large_storage_value(2), id="new_sibling"),
        pytest.param([1, 2], large_storage_value(9), id="modified_sibling"),
    ],
)
def test_witness_state_delete_with_dirty_sibling_omits_post_state_node(
    pre: Alloc,
    blockchain_test: BlockchainTestFiller,
    pre_slots: list[int],
    new_slot_2_value: int,
) -> None:
    """
    A sibling that becomes dirty before branch collapse should not leak.

    The witness still needs the pre-state delete proof for slot 1 and the
    pre-state proof for slot 2. When slot 2 is new, that is an absence
    proof, and the node created only after slot 2 is inserted during
    execution must not appear. When slot 2 already exists, the updated
    surviving child is dirty and must not be re-recorded as auxiliary.
    """
    pre_storage = build_large_storage(pre_slots)
    post_storage = {2: new_slot_2_value}
    proof_nodes = collect_storage_proof_nodes(pre_storage, [1, 2])
    post_state_only_nodes = collect_storage_post_state_only_nodes(
        pre_storage=pre_storage,
        post_storage=post_storage,
        slot=2,
        pre_state_reference_slots=[1, 2],
    )
    assert proof_nodes
    assert len(post_state_only_nodes) == 1

    contract = pre.deploy_contract(
        code=Op.SSTORE(2, post_storage[2]) + Op.SSTORE(1, 0) + Op.STOP,
        storage=as_storage(pre_storage),
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
                        nodes_present=proof_nodes,
                        nodes_absent=post_state_only_nodes,
                    )
                ),
            )
        ],
        post={
            sender: Account(nonce=1),
            contract: Account(storage=post_storage),
        },
    )


def test_witness_state_account_delete_branch_collapse_adds_auxiliary_node(
    pre: Alloc,
    blockchain_test: BlockchainTestFiller,
) -> None:
    """
    Deleting a pre-funded account from a two-account branch forces collapse.

    A factory deploys a contract with CREATE2 at a pre-funded address, and
    the initcode self-destructs, sending the balance back to the factory.
    The contract was created in the same transaction, so the account ends
    the transaction empty and is deleted from the account trie. The witness
    should contain both the account's proof and the auxiliary node needed to
    preserve its untouched sibling.
    """
    initcode = bytes(Op.SELFDESTRUCT(Op.CALLER))
    salt = 0
    factory = pre.deploy_contract(
        code=Op.MSTORE(0, Op.PUSH32(initcode))
        + Op.SSTORE(
            0,
            Op.CREATE2(
                value=0,
                offset=32 - len(initcode),
                size=len(initcode),
                salt=salt,
            ),
        )
        + Op.STOP,
    )
    sender = pre.fund_eoa()
    created = compute_create2_address(factory, salt, initcode)
    # The sibling's key shares more leading nibbles with the created
    # account's than any other account's does, so the two leaves hang off
    # one branch, and deleting the created account collapses it.
    sibling = find_account_ending_absence_proof(
        merge_with_amsterdam_pre_alloc(pre), created
    )
    pre.fund_address(sibling, 1)
    # CREATE2 can deploy over an account that holds only a balance.
    pre.fund_address(created, 1)

    full_alloc = merge_with_amsterdam_pre_alloc(pre)
    proof_nodes = collect_account_proof_nodes(full_alloc, [created])
    auxiliary_nodes = collect_account_delete_auxiliary_nodes(
        full_alloc, created
    )
    assert len(auxiliary_nodes) == 1

    tx = Transaction(sender=sender, to=factory, gas_limit=500_000)

    blockchain_test(
        pre=pre,
        blocks=[
            Block(
                txs=[tx],
                expected_execution_witness_state=(
                    ExecutionWitnessStateExpectation(
                        nodes_present=proof_nodes + auxiliary_nodes,
                    )
                ),
            )
        ],
        post={
            sender: Account(nonce=1),
            factory: Account(nonce=2, balance=1, storage={0: created}),
            created: Account.NONEXISTENT,
            sibling: Account(balance=1, nonce=0),
        },
    )
