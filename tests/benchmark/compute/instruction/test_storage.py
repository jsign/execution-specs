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


# Calldata sizes for mode dispatch:
# - Exec mode: 64 bytes (start_slot + count) - no padding for efficiency
# - Init mode: 65 bytes (start_slot + count + padding) - not benchmarked
INIT_CALLDATA_SIZE = 65
EXEC_CALLDATA_SIZE = 64


def _build_cold_storage_contract(
    execution_code_body: Bytecode,
    ends_with_revert: bool,
) -> Bytecode:
    """
    Build contract bytecode for cold storage benchmarking.

    Contract has two modes controlled by calldata size:
    - CALLDATASIZE == 65: Init mode - SSTORE(slot, slot) for each slot
    - CALLDATASIZE == 64: Exec mode - run execution_code_body for each slot

    Exec mode uses 64 bytes (no padding) for efficiency since it's benchmarked.
    Init mode uses 65 bytes (with padding) since setup speed is irrelevant.

    Both modes read (start_slot, count) from calldata and loop count times,
    incrementing the slot number each iteration.

    Stack layout during loops: [count, current_slot]
    """
    # Shared bytecode components
    loop_condition = (
        Op.PUSH1(1) + Op.SWAP1 + Op.SUB + Op.DUP1 + Op.ISZERO + Op.ISZERO
    )
    slot_increment = Op.SWAP1 + Op.PUSH1(1) + Op.ADD + Op.SWAP1
    calldata_load = Op.CALLDATALOAD(0) + Op.CALLDATALOAD(32)

    # Init loop: SSTORE(slot, slot), increment slot
    init_loop_body = Op.DUP2 + Op.DUP1 + Op.SSTORE + slot_increment

    # Exec loop: user-provided body, increment slot
    exec_loop_body = execution_code_body + slot_increment

    # Calculate bytecode offsets
    # Dispatch: CALLDATASIZE PUSH1(65) EQ PUSH2(init_offset) JUMPI = 8 bytes
    dispatch_len = 8
    exec_loop_target = dispatch_len + len(calldata_load)

    # Build execution code
    exec_suffix = Op.REVERT(0, 0) if ends_with_revert else Op.STOP
    exec_code = (
        calldata_load
        + Op.JUMPDEST
        + exec_loop_body
        + Op.JUMPI(exec_loop_target, loop_condition)
        + exec_suffix
    )

    # Build init code (placed after exec code)
    init_offset = dispatch_len + len(exec_code)
    # +1 for first JUMPDEST
    init_loop_target = init_offset + 1 + len(calldata_load)

    init_code = (
        Op.JUMPDEST
        + calldata_load
        + Op.JUMPDEST
        + init_loop_body
        + Op.JUMPI(init_loop_target, loop_condition)
        + Op.STOP
    )

    # Combined contract: dispatch + exec + init
    return (
        Op.CALLDATASIZE
        + Op.PUSH1(INIT_CALLDATA_SIZE)
        + Op.EQ
        + Op.PUSH2(init_offset)
        + Op.JUMPI
        + exec_code
        + init_code
    )


def _deploy_cold_storage_contract(
    pre: Alloc,
    fork: Fork,
    execution_code_body: Bytecode,
    ends_with_revert: bool,
    init_slot_count: int,
    tx_gas_limit: int,
) -> tuple[Address, list[Transaction]]:
    """
    Deploy cold storage contract and create init transactions.

    Returns (contract_address, setup_transactions).
    """
    gas_costs = fork.gas_costs()
    intrinsic_calc = fork.transaction_intrinsic_cost_calculator()

    contract_code = _build_cold_storage_contract(
        execution_code_body, ends_with_revert
    )

    # Deploy using EXTCODECOPY pattern
    code_holder = pre.deploy_contract(code=contract_code)
    creation_code = Op.EXTCODECOPY(
        code_holder, 0, 0, Op.EXTCODESIZE(code_holder)
    ) + Op.RETURN(0, Op.MSIZE)

    sender = pre.fund_eoa()
    contract_address = compute_create_address(address=sender, nonce=0)

    setup_txs: list[Transaction] = []
    with TestPhaseManager.setup():
        # Contract deployment tx
        setup_txs.append(
            Transaction(
                to=None,
                gas_limit=tx_gas_limit,
                data=creation_code,
                sender=sender,
            )
        )

    # Init transactions to populate storage slots
    if init_slot_count > 0:
        # Gas per init loop iteration
        init_loop_gas = (
            gas_costs.G_STORAGE_SET
            + gas_costs.G_COLD_SLOAD
            + gas_costs.G_JUMPDEST
            + gas_costs.G_VERY_LOW * 12  # DUPs, SWAPs, PUSHs, arithmetic
            + gas_costs.G_HIGH  # JUMPI
        )

        worst_intrinsic = intrinsic_calc(calldata=b"\xff" * INIT_CALLDATA_SIZE)
        max_slots_per_tx = (
            tx_gas_limit * 9 // 10 - worst_intrinsic
        ) // init_loop_gas

        num_init_txs = math.ceil(init_slot_count / max_slots_per_tx)
        for i in range(num_init_txs):
            start = 1 + i * max_slots_per_tx
            remaining = init_slot_count - i * max_slots_per_tx
            count = min(max_slots_per_tx, remaining)
            # Padding byte triggers init mode (65 bytes)
            calldata = (
                start.to_bytes(32, "big") + count.to_bytes(32, "big") + b"\x00"
            )

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
            StorageAction.READ, TransactionResult.SUCCESS, id="SSLOAD"
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
@pytest.mark.parametrize("absent_slots", [True, False])
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

    Tests SLOAD/SSTORE on cold storage slots. Each slot is accessed exactly
    once to ensure cold access costs. For forks with tx gas limit caps
    (e.g., Osaka), execution is split across multiple transactions with
    different slot ranges.
    """
    gas_costs = fork.gas_costs()
    intrinsic_gas_calc = fork.transaction_intrinsic_cost_calculator()

    # Define `execution_code_body` based on storage_action, and corresponding
    # gas costs per loop iteration.
    if storage_action == StorageAction.READ:
        storage_op_cost = 0
        execution_code_body = Op.POP(Op.SLOAD(Op.DUP2))
        bytecode_cost = gas_costs.G_VERY_LOW + gas_costs.G_BASE
    elif storage_action == StorageAction.WRITE_SAME_VALUE:
        if absent_slots:
            storage_op_cost = gas_costs.G_STORAGE_SET
        else:
            storage_op_cost = gas_costs.G_WARM_SLOAD
        execution_code_body = Op.DUP2 + Op.DUP1 + Op.SSTORE
        bytecode_cost = gas_costs.G_VERY_LOW * 2
    elif storage_action == StorageAction.WRITE_NEW_VALUE:
        if absent_slots:
            storage_op_cost = gas_costs.G_STORAGE_SET
        else:
            storage_op_cost = gas_costs.G_STORAGE_RESET
        execution_code_body = Op.DUP2 + Op.NOT(0) + Op.SWAP1 + Op.SSTORE
        bytecode_cost = gas_costs.G_VERY_LOW * 4

    # Costs related to loop control flow i.e. iterator condition checking.
    loop_overhead = (
        gas_costs.G_JUMPDEST
        # 4 (slot_increment: SWAP1, PUSH1, ADD, SWAP1)
        # + 6 (loop_condition: PUSH1, SWAP1, SUB, DUP1, ISZERO, ISZERO)
        # + 1 (PUSH2 for JUMPI target)
        + gas_costs.G_VERY_LOW * 11
        + gas_costs.G_HIGH  # JUMPI
    )

    loop_cost = (
        gas_costs.G_COLD_SLOAD
        + storage_op_cost
        + bytecode_cost
        + loop_overhead
    )

    # The attack contract has dispatch logic at the start. Depending on
    # the CALLDATASIZE, it jumps to either init or execution mode.
    #
    # init mode is only used for the setup phase to initialize storage slots.
    # execution mode is used for the benchmarked storage accesses.
    #
    # This separation is required since we can't do storage initialization
    # in contract creation code due to potential transaction gas limits.
    prefix_cost = (
        gas_costs.G_BASE  # CALLDATASIZE
        + gas_costs.G_VERY_LOW * 3  # PUSH1 + EQ + PUSH2
        + gas_costs.G_HIGH  # JUMPI
        + gas_costs.G_VERY_LOW * 4  # 2x (PUSH1 + CALLDATALOAD)
    )

    # Suffix cost (REVERT only): REVERT(0, 0)
    if tx_result == TransactionResult.REVERT:
        suffix_cost = gas_costs.G_VERY_LOW * 2
    else:
        suffix_cost = 0

    # The attack txs use calldata to signal start_slot and count.
    # Since calldata gas cost varies based on zero vs non-zero bytes,
    # we use worst-case intrinsic gas (all non-zero bytes) for calculations.
    #
    # Exec calldata: [start_slot (32B), count (32B)] = 64 bytes
    # Worst case: 64 bytes * 16 gas = 1024 gas for calldata.
    # Gas difference between average and worst case is negligible.
    worst_intrinsic = intrinsic_gas_calc(
        calldata=b"\xff" * EXEC_CALLDATA_SIZE,
        return_cost_deducted_prior_execution=True,
    )

    # How many slots can we access with gas_benchmark_value?
    num_target_slots = (
        gas_benchmark_value - worst_intrinsic - prefix_cost - suffix_cost
    ) // loop_cost

    contract_address, setup_txs = _deploy_cold_storage_contract(
        pre=pre,
        fork=fork,
        execution_code_body=execution_code_body,
        ends_with_revert=(tx_result == TransactionResult.REVERT),
        init_slot_count=num_target_slots if not absent_slots else 0,
        tx_gas_limit=tx_gas_limit,
    )

    # Calculate max slots per tx
    available_gas = tx_gas_limit - worst_intrinsic - prefix_cost
    max_slots_per_tx = available_gas // loop_cost

    num_exec_txs = math.ceil(num_target_slots / max_slots_per_tx)
    exec_txs: list[Transaction] = []
    total_gas_used = 0

    for tx_index in range(num_exec_txs):
        start_slot = 1 + tx_index * max_slots_per_tx

        slots_in_tx = min(
            max_slots_per_tx, num_target_slots - tx_index * max_slots_per_tx
        )

        calldata = start_slot.to_bytes(32, "big") + slots_in_tx.to_bytes(
            32, "big"
        )
        tx_intrinsic = intrinsic_gas_calc(
            calldata=calldata, return_cost_deducted_prior_execution=True
        )

        # Calculate gas limit for this transaction.
        # For OOG: give gas for (slots-1) to trigger OOG on last iteration.
        # Each tx runs out of gas, similar to how REVERT makes each tx revert.
        if tx_result == TransactionResult.OUT_OF_GAS:
            slots_for_gas = slots_in_tx - 1
        else:
            slots_for_gas = slots_in_tx

        tx_gas = tx_intrinsic + prefix_cost + loop_cost * slots_for_gas
        is_last_tx = tx_index == num_exec_txs - 1
        if is_last_tx and tx_result == TransactionResult.REVERT:
            tx_gas += suffix_cost  # The REVERT(0, 0)

        total_gas_used += tx_gas

        exec_txs.append(
            Transaction(
                to=contract_address,
                gas_limit=tx_gas,
                data=calldata,
                sender=pre.fund_eoa(),
            )
        )

    # Determine expected storage values after execution.
    # Recall init phase writes slot[i] = i for all slots.
    post = {}
    if not absent_slots:
        if tx_result == TransactionResult.SUCCESS:
            slots_with_exec_values = num_target_slots
        else:
            # REVERT and OUT_OF_GAS: all txs revert, no exec values persist
            slots_with_exec_values = 0

        if storage_action == StorageAction.WRITE_NEW_VALUE:
            # Executed slots have -1, remaining keep init value (slot[i] = i)
            storage = {
                i: (2**256 - 1 if i <= slots_with_exec_values else i)
                for i in range(1, num_target_slots + 1)
            }
        else:
            # READ or WRITE_SAME_VALUE: all slots have init value
            storage = {i: i for i in range(1, num_target_slots + 1)}

        post = {contract_address: Account(storage=storage)}

    blocks = [Block(txs=setup_txs)]
    with TestPhaseManager.execution():
        blocks.append(Block(txs=exec_txs))

    benchmark_test(
        blocks=blocks,
        expected_benchmark_gas_used=total_gas_used,
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
    gas_benchmark_value: int,
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
