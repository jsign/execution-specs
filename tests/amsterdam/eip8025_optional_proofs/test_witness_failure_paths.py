"""
Witness-focused failure-path coverage for EIP-8025 optional proofs.

These cases target border behavior where execution fails, but witness capture
must still reflect the exact point where state/code access happened (or did
not happen yet).
"""

import pytest
from execution_testing import (
    Account,
    Alloc,
    Block,
    BlockchainTestFiller,
    Bytecode,
    Op,
    Transaction,
)

pytestmark = pytest.mark.valid_at("Osaka")

REFERENCE_SPEC_GIT_PATH = "TODO"
REFERENCE_SPEC_VERSION = "TODO"


CALL_ARGS_SIZE = 2**20
# Calibrated Osaka boundary for this exact CALL+SSTORE sequence:
# - `CALL_PRECHECK_OOG_GAS_LIMIT` keeps target/delegation path out of witness.
# - `+1` gas crosses the boundary and includes it.
CALL_PRECHECK_OOG_GAS_LIMIT = 2_219_076
CALL_PRECHECK_PLUS_ONE_GAS_LIMIT = CALL_PRECHECK_OOG_GAS_LIMIT + 1


def _push_spam(push_count: int) -> Bytecode:
    """Build bytecode with `push_count` PUSH1(0) instructions."""
    code = Bytecode()
    for _ in range(push_count):
        code += Op.PUSH1(0)
    return code


def test_witness_oog_before_call_access(
    pre: Alloc,
    blockchain_test: BlockchainTestFiller,
) -> None:
    """
    CALL runs out of gas before it can perform account/code access.

    The large calldata size forces memory expansion OOG in CALL setup, so any
    witness account/code capture for the call target would be incorrect.
    """
    sender = pre.fund_eoa()
    target = pre.deploy_contract(Op.STOP)

    caller = pre.deploy_contract(
        code=(
            Op.CALL(gas=0, address=target, args_size=CALL_ARGS_SIZE)
            + Op.SSTORE(0, 1)
            + Op.STOP
        ),
        storage={},
    )

    tx = Transaction(
        sender=sender,
        to=caller,
        gas_limit=CALL_PRECHECK_OOG_GAS_LIMIT,
    )

    # The transaction reverts on OOG, so contract storage remains unchanged.
    blockchain_test(
        pre=pre,
        blocks=[Block(txs=[tx])],
        post={caller: Account(storage={})},
    )


def test_witness_oog_before_call_access_plus_one_gas(
    pre: Alloc,
    blockchain_test: BlockchainTestFiller,
) -> None:
    """
    Same CALL setup as `test_witness_oog_before_call_access`, but with +1 gas.

    This crosses the exact setup-OOG boundary: account/code access for target
    can now happen, so witness is expected to include target state/code.
    """
    sender = pre.fund_eoa()
    target = pre.deploy_contract(Op.STOP)

    caller = pre.deploy_contract(
        code=(
            Op.CALL(gas=0, address=target, args_size=CALL_ARGS_SIZE)
            + Op.SSTORE(0, 1)
            + Op.STOP
        ),
        storage={},
    )

    tx = Transaction(
        sender=sender,
        to=caller,
        gas_limit=CALL_PRECHECK_PLUS_ONE_GAS_LIMIT,
    )

    blockchain_test(
        pre=pre,
        blocks=[Block(txs=[tx])],
        post={caller: Account(storage={})},
    )


def test_witness_oog_before_delegation_access(
    pre: Alloc,
    blockchain_test: BlockchainTestFiller,
) -> None:
    """
    CALL to a delegated EOA runs out of gas before delegation lookup.

    This guards against over-capturing pointer/delegated bytecode when CALL
    cannot even pay the pre-call gas requirements.
    """
    sender = pre.fund_eoa()
    delegated_impl = pre.deploy_contract(Op.STOP)
    delegated_pointer = pre.fund_eoa(delegation=delegated_impl)

    caller = pre.deploy_contract(
        code=(
            Op.CALL(gas=0, address=delegated_pointer, args_size=CALL_ARGS_SIZE)
            + Op.SSTORE(0, 1)
            + Op.STOP
        ),
        storage={},
    )

    tx = Transaction(
        sender=sender,
        to=caller,
        gas_limit=CALL_PRECHECK_OOG_GAS_LIMIT,
    )

    blockchain_test(
        pre=pre,
        blocks=[Block(txs=[tx])],
        post={caller: Account(storage={})},
    )


def test_witness_oog_before_delegation_access_plus_one_gas(
    pre: Alloc,
    blockchain_test: BlockchainTestFiller,
) -> None:
    """
    Same delegated CALL setup as `test_witness_oog_before_delegation_access`,
    but with +1 gas.

    This crosses the setup-OOG boundary and should allow delegation-path
    access capture in witness.
    """
    sender = pre.fund_eoa()
    delegated_impl = pre.deploy_contract(Op.STOP)
    delegated_pointer = pre.fund_eoa(delegation=delegated_impl)

    caller = pre.deploy_contract(
        code=(
            Op.CALL(gas=0, address=delegated_pointer, args_size=CALL_ARGS_SIZE)
            + Op.SSTORE(0, 1)
            + Op.STOP
        ),
        storage={},
    )

    tx = Transaction(
        sender=sender,
        to=caller,
        gas_limit=CALL_PRECHECK_PLUS_ONE_GAS_LIMIT,
    )

    blockchain_test(
        pre=pre,
        blocks=[Block(txs=[tx])],
        post={caller: Account(storage={})},
    )


def test_witness_subcall_oog_after_access(
    pre: Alloc,
    blockchain_test: BlockchainTestFiller,
) -> None:
    """
    Subcall fails with OOG after target access and code lookup already
    happened.

    Parent execution continues and stores the failed CALL return code (`0`).
    """
    sender = pre.fund_eoa()
    target = pre.deploy_contract(Op.SSTORE(0, 1) + Op.STOP)

    caller = pre.deploy_contract(
        code=Op.SSTORE(0, Op.CALL(gas=0, address=target)) + Op.STOP,
        storage={},
    )

    tx = Transaction(sender=sender, to=caller, gas_limit=300_000)

    blockchain_test(
        pre=pre,
        blocks=[Block(txs=[tx])],
        post={caller: Account(storage={0: 0})},
    )


def test_witness_call_insufficient_balance_after_access(
    pre: Alloc,
    blockchain_test: BlockchainTestFiller,
) -> None:
    """
    CALL fails for insufficient caller balance after target access path runs.

    Target access/delegation checks happen before value-transfer balance check,
    so witness capture must reflect that ordering.
    """
    sender = pre.fund_eoa()
    target = pre.deploy_contract(Op.STOP)

    caller = pre.deploy_contract(
        code=Op.SSTORE(0, Op.CALL(gas=100_000, address=target, value=1))
        + Op.STOP,
        storage={},
    )

    tx = Transaction(sender=sender, to=caller, gas_limit=300_000)

    blockchain_test(
        pre=pre,
        blocks=[Block(txs=[tx])],
        post={caller: Account(storage={0: 0})},
    )


def test_witness_revert_before_state_access(
    pre: Alloc,
    blockchain_test: BlockchainTestFiller,
) -> None:
    """
    REVERT executes before BALANCE is reached.

    This checks that unreachable state-access opcodes do not leak into witness
    account capture.
    """
    sender = pre.fund_eoa()
    balance_target = pre.fund_eoa()

    contract = pre.deploy_contract(
        code=Op.REVERT(0, 0) + Op.BALANCE(balance_target) + Op.STOP
    )

    tx = Transaction(sender=sender, to=contract, gas_limit=200_000)

    blockchain_test(
        pre=pre,
        blocks=[Block(txs=[tx])],
        post={},
    )


def test_witness_revert_after_state_access(
    pre: Alloc,
    blockchain_test: BlockchainTestFiller,
) -> None:
    """
    BALANCE executes first, then REVERT aborts transaction.

    Access that happened before REVERT should still be represented in witness.
    """
    sender = pre.fund_eoa()
    balance_target = pre.fund_eoa()

    contract = pre.deploy_contract(code=Op.BALANCE(balance_target) + Op.REVERT(0, 0))

    tx = Transaction(sender=sender, to=contract, gas_limit=200_000)

    blockchain_test(
        pre=pre,
        blocks=[Block(txs=[tx])],
        post={},
    )


def test_witness_stack_overflow_before_state_access(
    pre: Alloc,
    blockchain_test: BlockchainTestFiller,
) -> None:
    """
    Stack overflow happens before BALANCE executes.

    This checks that failed execution does not capture the BALANCE target as
    an accessed account when overflow aborts first.
    """
    sender = pre.fund_eoa()
    balance_target = pre.fund_eoa()

    contract = pre.deploy_contract(
        code=_push_spam(1025) + Op.BALANCE(balance_target) + Op.STOP
    )

    tx = Transaction(sender=sender, to=contract, gas_limit=300_000)

    blockchain_test(
        pre=pre,
        blocks=[Block(txs=[tx])],
        post={},
    )


def test_witness_stack_overflow_after_state_access(
    pre: Alloc,
    blockchain_test: BlockchainTestFiller,
) -> None:
    """
    BALANCE executes first, then stack overflow aborts execution.

    This is the opposite edge of the previous case: account access happens,
    then failure occurs later.
    """
    sender = pre.fund_eoa()
    balance_target = pre.fund_eoa()

    contract = pre.deploy_contract(
        code=Op.BALANCE(balance_target) + _push_spam(1024) + Op.STOP
    )

    tx = Transaction(sender=sender, to=contract, gas_limit=300_000)

    blockchain_test(
        pre=pre,
        blocks=[Block(txs=[tx])],
        post={},
    )
