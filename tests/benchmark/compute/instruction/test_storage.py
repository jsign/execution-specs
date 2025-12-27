"""
Benchmark storage instructions.

Supported Opcodes:
- SLOAD
- SSTORE
- TLOAD
- TSTORE
"""

import math

import pytest
from execution_testing import (
    Account,
    Address,
    Alloc,
    BenchmarkTestFiller,
    Block,
    Bytecode,
    Environment,
    ExtCallGenerator,
    Fork,
    JumpLoopGenerator,
    Op,
    TestPhaseManager,
    Transaction,
    While,
    compute_create_address,
)

from tests.benchmark.compute.helpers import StorageAction, TransactionResult


@pytest.mark.repricing(fixed_key=False, fixed_value=False)
@pytest.mark.parametrize("fixed_key", [True, False])
@pytest.mark.parametrize("fixed_value", [True, False])
def test_tload(
    benchmark_test: BenchmarkTestFiller,
    fixed_key: bool,
    fixed_value: bool,
) -> None:
    """Benchmark TLOAD instruction."""
    setup = Bytecode()
    if not fixed_key and not fixed_value:
        setup = Op.GAS + Op.TSTORE(Op.DUP2, Op.GAS)
        attack_block = Op.TLOAD(Op.DUP1)
    if not fixed_key and fixed_value:
        attack_block = Op.TLOAD(Op.GAS)
    if fixed_key and not fixed_value:
        setup = Op.TSTORE(Op.CALLDATASIZE, Op.GAS)
        attack_block = Op.TLOAD(Op.CALLDATASIZE)
    if fixed_key and fixed_value:
        attack_block = Op.TLOAD(Op.CALLDATASIZE)

    tx_data = b"42" if fixed_key and not fixed_value else b""

    benchmark_test(
        code_generator=ExtCallGenerator(
            setup=setup,
            attack_block=attack_block,
            tx_kwargs={"data": tx_data},
        ),
    )


@pytest.mark.repricing(fixed_key=False, fixed_value=False)
@pytest.mark.parametrize("fixed_key", [True, False])
@pytest.mark.parametrize("fixed_value", [True, False])
def test_tstore(
    benchmark_test: BenchmarkTestFiller,
    fixed_key: bool,
    fixed_value: bool,
) -> None:
    """Benchmark TSTORE instruction."""
    init_key = 42
    setup = Op.PUSH1(init_key)

    # If fixed_value is False, we use GAS as a cheap way of always
    # storing a different value than the previous one.
    attack_block = Op.TSTORE(Op.DUP2, Op.GAS if not fixed_value else Op.DUP1)

    # If fixed_key is False, we mutate the key on every iteration of the
    # big loop.
    cleanup = Op.POP + Op.GAS if not fixed_key else Bytecode()

    benchmark_test(
        code_generator=JumpLoopGenerator(
            setup=setup, attack_block=attack_block, cleanup=cleanup
        ),
    )


def _setup_cold_storage_contract(
    pre: Alloc,
    fork: Fork,
    num_target_slots: int,
    init_slots: bool,
    execution_code_body: Bytecode,
    tx_result: TransactionResult,
    tx_gas_limit: int,
) -> tuple[Address, list[Transaction]]:
    """
    Deploy a contract for cold storage benchmarking and optionally initialize slots.

    The contract has two execution paths:
    - If calldata present: run init loop to initialize slots from (start_slot, count)
    - If no calldata: run the execution code

    Returns the contract address and list of setup transactions.
    """
    gas_costs = fork.gas_costs()
    intrinsic_gas_cost_calc = fork.transaction_intrinsic_cost_calculator()

    # Common loop condition: decrement counter, check if non-zero
    loop_condition = Op.PUSH1(1) + Op.SWAP1 + Op.SUB + Op.DUP1 + Op.ISZERO + Op.ISZERO

    # Init loop code: reads (start_slot, count) from calldata
    init_loop = (
        Op.CALLDATALOAD(0)  # start_slot
        + Op.CALLDATALOAD(32)  # count
        + While(
            body=(
                Op.DUP2  # [start_slot, count, start_slot]
                + Op.DUP1  # [start_slot, start_slot, ...]
                + Op.SSTORE  # [count, start_slot]
                + Op.SWAP1  # [start_slot, count]
                + Op.PUSH1(1)
                + Op.ADD  # [start_slot+1, count]
                + Op.SWAP1  # [count, start_slot+1]
            ),
            condition=loop_condition,
        )
        + Op.STOP
    )

    # Combined contract: init check + init code + execution code
    # Prefix: CALLDATASIZE + ISZERO + PUSH2 + JUMPI = 6 bytes
    prefix_len = 6
    execution_offset = prefix_len + len(init_loop)

    # Execution code with adjusted jump target for embedding
    execution_code_start = prefix_len + len(init_loop) + 1  # +1 for outer JUMPDEST
    code_prefix = Op.PUSH4(num_target_slots) + Op.JUMPDEST
    jump_target = execution_code_start + len(code_prefix) - 1
    code_loop = execution_code_body + Op.JUMPI(jump_target, loop_condition)
    execution_code = code_prefix + code_loop
    if tx_result == TransactionResult.REVERT:
        execution_code += Op.REVERT(0, 0)
    else:
        execution_code += Op.STOP

    contract_code = (
        Op.CALLDATASIZE
        + Op.ISZERO
        + Op.PUSH2(execution_offset)
        + Op.JUMPI
        + init_loop
        + Op.JUMPDEST  # execution_start
        + execution_code
    )

    # Deploy via EXTCODECOPY pattern
    contract_code_address = pre.deploy_contract(code=contract_code)
    creation_code = (
        Op.EXTCODECOPY(
            address=contract_code_address,
            dest_offset=0,
            offset=0,
            size=Op.EXTCODESIZE(contract_code_address),
        )
        + Op.RETURN(0, Op.MSIZE)
    )

    sender_addr = pre.fund_eoa()
    contract_address = compute_create_address(address=sender_addr, nonce=0)
    setup_txs: list[Transaction] = []

    with TestPhaseManager.setup():
        setup_txs.append(
            Transaction(
                to=None,
                gas_limit=tx_gas_limit,
                data=creation_code,
                sender=sender_addr,
            )
        )

    # Create init transactions to initialize slots
    if init_slots:
        slot_init_loop_cost = (
            gas_costs.G_STORAGE_SET
            + gas_costs.G_COLD_SLOAD
            + gas_costs.G_JUMPDEST
            + gas_costs.G_VERY_LOW * 12  # DUPs, SWAPs, PUSHs, SUB, ADD, ISZEROs
            + gas_costs.G_HIGH
        )

        max_slots_per_tx = (
            tx_gas_limit * 9 // 10 - intrinsic_gas_cost_calc()
        ) // slot_init_loop_cost

        for i in range(math.ceil(num_target_slots / max_slots_per_tx)):
            start_slot = 1 + i * max_slots_per_tx
            count = min(max_slots_per_tx, num_target_slots - i * max_slots_per_tx)
            calldata = start_slot.to_bytes(32, "big") + count.to_bytes(32, "big")

            setup_txs.append(
                Transaction(
                    to=contract_address,
                    gas_limit=tx_gas_limit,
                    data=calldata,
                    sender=pre.fund_eoa(),
                )
            )

    return contract_address, setup_txs


@pytest.mark.parametrize(
    "storage_action,tx_result",
    [
        pytest.param(
            StorageAction.READ,
            TransactionResult.SUCCESS,
            id="SSLOAD",
        ),
        pytest.param(
            StorageAction.WRITE_SAME_VALUE,
            TransactionResult.SUCCESS,
            id="SSTORE same value",
        ),
        pytest.param(
            StorageAction.WRITE_SAME_VALUE,
            TransactionResult.REVERT,
            id="SSTORE same value, revert",
        ),
        pytest.param(
            StorageAction.WRITE_SAME_VALUE,
            TransactionResult.OUT_OF_GAS,
            id="SSTORE same value, out of gas",
        ),
        pytest.param(
            StorageAction.WRITE_NEW_VALUE,
            TransactionResult.SUCCESS,
            id="SSTORE new value",
        ),
        pytest.param(
            StorageAction.WRITE_NEW_VALUE,
            TransactionResult.REVERT,
            id="SSTORE new value, revert",
        ),
        pytest.param(
            StorageAction.WRITE_NEW_VALUE,
            TransactionResult.OUT_OF_GAS,
            id="SSTORE new value, out of gas",
        ),
    ],
)
@pytest.mark.parametrize(
    "absent_slots",
    [
        True,
        False,
    ],
)
def test_storage_access_cold(
    benchmark_test: BenchmarkTestFiller,
    pre: Alloc,
    fork: Fork,
    storage_action: StorageAction,
    absent_slots: bool,
    gas_benchmark_value: int,
    tx_gas_limit: int,
    tx_result: TransactionResult,
) -> None:
    """
    Benchmark cold storage slot accesses.
    """
    gas_costs = fork.gas_costs()
    intrinsic_gas_cost_calc = fork.transaction_intrinsic_cost_calculator()

    # Calculate loop cost based on storage action
    loop_cost = gas_costs.G_COLD_SLOAD  # All accesses are always cold
    if storage_action == StorageAction.WRITE_NEW_VALUE:
        loop_cost += gas_costs.G_STORAGE_RESET if not absent_slots else gas_costs.G_STORAGE_SET
    elif storage_action == StorageAction.WRITE_SAME_VALUE:
        loop_cost += gas_costs.G_STORAGE_SET if absent_slots else gas_costs.G_WARM_SLOAD

    # Build execution code body based on storage action
    execution_code_body = Bytecode()
    if storage_action == StorageAction.WRITE_SAME_VALUE:
        execution_code_body = Op.SSTORE(Op.DUP1, Op.DUP1)
        loop_cost += gas_costs.G_VERY_LOW * 2
    elif storage_action == StorageAction.WRITE_NEW_VALUE:
        execution_code_body = Op.SSTORE(Op.DUP2, Op.NOT(0))
        loop_cost += gas_costs.G_VERY_LOW * 3
    elif storage_action == StorageAction.READ:
        execution_code_body = Op.POP(Op.SLOAD(Op.DUP1))
        loop_cost += gas_costs.G_VERY_LOW + gas_costs.G_BASE

    # Add jump-logic costs
    loop_cost += (
        gas_costs.G_JUMPDEST
        + gas_costs.G_VERY_LOW * 7  # ISZEROs, PUSHs, SWAPs, SUB, DUP
        + gas_costs.G_HIGH
    )

    prefix_cost = (
        gas_costs.G_VERY_LOW  # Target slots push
        + gas_costs.G_BASE  # CALLDATASIZE
        + gas_costs.G_VERY_LOW * 2  # ISZERO, PUSH2
        + gas_costs.G_HIGH  # JUMPI
        + gas_costs.G_JUMPDEST  # outer JUMPDEST
    )

    suffix_cost = gas_costs.G_VERY_LOW * 2 if tx_result == TransactionResult.REVERT else 0

    num_target_slots = (
        gas_benchmark_value - intrinsic_gas_cost_calc() - prefix_cost - suffix_cost
    ) // loop_cost
    if tx_result == TransactionResult.OUT_OF_GAS:
        num_target_slots += 1

    total_gas_used = (
        num_target_slots * loop_cost + intrinsic_gas_cost_calc() + prefix_cost + suffix_cost
    )

    # Setup: deploy contract and initialize slots
    contract_address, setup_txs = _setup_cold_storage_contract(
        pre=pre,
        fork=fork,
        num_target_slots=num_target_slots,
        init_slots=not absent_slots,
        execution_code_body=execution_code_body,
        tx_result=tx_result,
        tx_gas_limit=tx_gas_limit,
    )

    # Execution phase
    blocks = [Block(txs=setup_txs)]
    with TestPhaseManager.execution():
        blocks.append(
            Block(
                txs=[
                    Transaction(
                        to=contract_address,
                        gas_limit=gas_benchmark_value,
                        sender=pre.fund_eoa(),
                    )
                ]
            )
        )

    # Post check
    post = {}
    if not absent_slots:
        if storage_action == StorageAction.WRITE_NEW_VALUE and tx_result == TransactionResult.SUCCESS:
            storage = {i: 2**256 - 1 for i in range(1, num_target_slots + 1)}
        else:
            storage = {i: i for i in range(1, num_target_slots + 1)}
        post = {contract_address: Account(storage=storage)}

    benchmark_test(
        blocks=blocks,
        expected_benchmark_gas_used=(
            total_gas_used if tx_result != TransactionResult.OUT_OF_GAS else gas_benchmark_value
        ),
        post=post,
    )


@pytest.mark.parametrize(
    "storage_action",
    [
        pytest.param(StorageAction.READ, id="SLOAD"),
        pytest.param(StorageAction.WRITE_SAME_VALUE, id="SSTORE same value"),
        pytest.param(StorageAction.WRITE_NEW_VALUE, id="SSTORE new value"),
    ],
)
def test_storage_access_warm(
    benchmark_test: BenchmarkTestFiller,
    pre: Alloc,
    storage_action: StorageAction,
    fork: Fork,
    gas_benchmark_value: int,
    env: Environment,
    tx_gas_limit: int,
) -> None:
    """
    Benchmark warm storage slot accesses.
    """
    blocks = []

    # The warm access is done in storage slot 0.

    # Contract code
    execution_code_body = Bytecode()
    if storage_action == StorageAction.WRITE_SAME_VALUE:
        execution_code_body = Op.SSTORE(0, Op.DUP1)
    elif storage_action == StorageAction.WRITE_NEW_VALUE:
        execution_code_body = Op.SSTORE(0, Op.GAS)
    elif storage_action == StorageAction.READ:
        execution_code_body = Op.POP(Op.SLOAD(0))

    execution_code = Op.SLOAD(0) + While(
        body=execution_code_body,
    )
    execution_code_address = pre.deploy_contract(code=execution_code)

    creation_code = (
        Op.SSTORE(0, 42)
        + Op.EXTCODECOPY(
            address=execution_code_address,
            dest_offset=0,
            offset=0,
            size=Op.EXTCODESIZE(execution_code_address),
        )
        + Op.RETURN(0, Op.MSIZE)
    )

    with TestPhaseManager.setup():
        sender_addr = pre.fund_eoa()
        setup_tx = Transaction(
            to=None,
            gas_limit=tx_gas_limit,
            data=creation_code,
            sender=sender_addr,
        )
        blocks.append(Block(txs=[setup_tx]))

    contract_address = compute_create_address(address=sender_addr, nonce=0)

    with TestPhaseManager.execution():
        num_exec_txs = math.ceil(gas_benchmark_value / tx_gas_limit)
        txs = []
        for i in range(num_exec_txs):
            gas_limit = min(
                tx_gas_limit, gas_benchmark_value - i * tx_gas_limit
            )
            op_tx = Transaction(
                to=contract_address,
                gas_limit=gas_limit,
                sender=pre.fund_eoa(),
            )
            txs.append(op_tx)
        blocks.append(Block(txs=txs))

    benchmark_test(blocks=blocks)
