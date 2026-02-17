"""
Witness-focused ancestor coverage for EIP-8025 optional proofs.

These tests exercise BLOCKHASH/system-access combinations that determine the
ancestor header range included in execution witness.
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


def test_witness_ancestors_parent_only_baseline(
    pre: Alloc,
    blockchain_test: BlockchainTestFiller,
) -> None:
    """
    No explicit BLOCKHASH opcode in user tx.

    Witness ancestors should still include parent-header access induced by
    block-level system processing.
    """
    sender = pre.fund_eoa()
    contract = pre.deploy_contract(Op.SSTORE(0, 1) + Op.STOP)

    tx = Transaction(sender=sender, to=contract, gas_limit=150_000)

    blockchain_test(
        pre=pre,
        blocks=[Block(txs=[tx])],
        post={contract: Account(storage={0: 1})},
    )


def test_witness_ancestors_oldest_blockhash_extends_range(
    pre: Alloc,
    blockchain_test: BlockchainTestFiller,
) -> None:
    """
    Explicitly access older blocks in the BLOCKHASH window.

    Final block number is 6, so BLOCKHASH(1) and BLOCKHASH(4) must both be
    non-zero and should extend witness ancestor collection down to block 1.
    """
    sender = pre.fund_eoa()
    checker = pre.deploy_contract(
        code=(
            Op.SSTORE(0, Op.ISZERO(Op.BLOCKHASH(1)))
            + Op.SSTORE(1, Op.ISZERO(Op.BLOCKHASH(4)))
            + Op.STOP
        ),
        storage={},
    )

    tx = Transaction(sender=sender, to=checker, gas_limit=300_000)
    blocks = [Block(txs=[]) for _ in range(5)] + [Block(txs=[tx])]

    blockchain_test(
        pre=pre,
        blocks=blocks,
        post={checker: Account(storage={0: 0, 1: 0})},
    )


def test_witness_ancestors_current_and_future_blockhash_do_not_extend_range(
    pre: Alloc,
    blockchain_test: BlockchainTestFiller,
) -> None:
    """
    BLOCKHASH on current/future blocks must return zero.

    Final block number is 3 in this test, so both BLOCKHASH(3) and
    BLOCKHASH(4) are zero-valued.
    """
    sender = pre.fund_eoa()
    checker = pre.deploy_contract(
        code=(
            Op.SSTORE(0, Op.ISZERO(Op.BLOCKHASH(3)))
            + Op.SSTORE(1, Op.ISZERO(Op.BLOCKHASH(4)))
            + Op.STOP
        ),
        storage={},
    )

    tx = Transaction(sender=sender, to=checker, gas_limit=300_000)
    blocks = [Block(txs=[]) for _ in range(2)] + [Block(txs=[tx])]

    blockchain_test(
        pre=pre,
        blocks=blocks,
        post={checker: Account(storage={0: 1, 1: 1})},
    )


def test_witness_ancestors_oldest_of_multiple_accesses(
    pre: Alloc,
    blockchain_test: BlockchainTestFiller,
) -> None:
    """
    Multiple in-range BLOCKHASH reads should pick the oldest block as anchor.

    Final block number is 8; all checks below are in-range and must be
    non-zero. Ancestor selection should therefore anchor at block 2.
    """
    sender = pre.fund_eoa()
    checker = pre.deploy_contract(
        code=(
            Op.SSTORE(0, Op.ISZERO(Op.BLOCKHASH(7)))
            + Op.SSTORE(1, Op.ISZERO(Op.BLOCKHASH(6)))
            + Op.SSTORE(2, Op.ISZERO(Op.BLOCKHASH(2)))
            + Op.STOP
        ),
        storage={},
    )

    tx = Transaction(sender=sender, to=checker, gas_limit=400_000)
    blocks = [Block(txs=[]) for _ in range(7)] + [Block(txs=[tx])]

    blockchain_test(
        pre=pre,
        blocks=blocks,
        post={checker: Account(storage={0: 0, 1: 0, 2: 0})},
    )


def test_witness_ancestors_blockhash_window_boundary(
    pre: Alloc,
    blockchain_test: BlockchainTestFiller,
) -> None:
    """
    Exercise the exact 256-block BLOCKHASH lower bound.

    Final block number is 257:
    - BLOCKHASH(1) is still in-range and must be non-zero.
    - BLOCKHASH(0) is out-of-range and must be zero.
    """
    sender = pre.fund_eoa()
    checker = pre.deploy_contract(
        code=(
            Op.SSTORE(0, Op.ISZERO(Op.BLOCKHASH(1)))
            + Op.SSTORE(1, Op.ISZERO(Op.BLOCKHASH(0)))
            + Op.STOP
        ),
        storage={},
    )

    tx = Transaction(sender=sender, to=checker, gas_limit=400_000)
    blocks = [Block(txs=[]) for _ in range(256)] + [Block(txs=[tx])]

    blockchain_test(
        pre=pre,
        blocks=blocks,
        post={checker: Account(storage={0: 0, 1: 1})},
    )
