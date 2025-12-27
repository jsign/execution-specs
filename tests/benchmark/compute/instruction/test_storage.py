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


def _setup_cold_storage_contract(
    pre: Alloc,
    fork: Fork,
    init_slots_count: int,
    execution_code_body: Bytecode,
    tx_result: TransactionResult,
    tx_gas_limit: int,
) -> tuple[Address, list[Transaction]]:
    """
    Deploy a contract for cold storage benchmarking.

    The contract has two execution paths controlled by calldata size:
    - CALLDATASIZE == 64: run init loop (setup phase)
    - CALLDATASIZE != 64: run execution loop (benchmark phase, uses 65 bytes)

    Both paths read (start_slot, count) from calldata[0:64] and loop through slots.
    Execution uses 65 bytes (64 + 1 padding) to distinguish from init's 64 bytes.
    Returns the contract address and list of setup transactions.
    """
    gas_costs = fork.gas_costs()
    intrinsic_gas_cost_calc = fork.transaction_intrinsic_cost_calculator()

    # Common loop condition: decrement counter, check if non-zero
    loop_condition = (
        Op.PUSH1(1) + Op.SWAP1 + Op.SUB + Op.DUP1 + Op.ISZERO + Op.ISZERO
    )

    # Slot increment: SWAP1 PUSH1(1) ADD SWAP1
    # Stack: [count, start_slot] → [count, start_slot+1]
    slot_increment = Op.SWAP1 + Op.PUSH1(1) + Op.ADD + Op.SWAP1

    # Init loop body: SSTORE(slot, slot), then increment slot
    init_loop_body = (
        Op.DUP2  # [start_slot, count, start_slot]
        + Op.DUP1  # [start_slot, start_slot, count, start_slot]
        + Op.SSTORE  # storage[start_slot] = start_slot; [count, start_slot]
        + slot_increment
    )

    # Execution loop body: action + increment slot
    execution_loop_body = execution_code_body + slot_increment

    # Build execution code (reads calldata, loops)
    # Dispatch prefix: CALLDATASIZE PUSH1(64) EQ PUSH2(init_offset) JUMPI = 8 bytes
    dispatch_prefix_len = 8

    # Execution code layout:
    # CALLDATALOAD(0) CALLDATALOAD(32) JUMPDEST <loop_body> JUMPI <suffix>
    calldata_load = Op.CALLDATALOAD(0) + Op.CALLDATALOAD(32)  # 6 bytes
    execution_loop_target = dispatch_prefix_len + len(calldata_load)

    execution_loop = (
        Op.JUMPDEST
        + execution_loop_body
        + Op.JUMPI(execution_loop_target, loop_condition)
    )
    if tx_result == TransactionResult.REVERT:
        execution_code = calldata_load + execution_loop + Op.REVERT(0, 0)
    else:
        execution_code = calldata_load + execution_loop + Op.STOP

    # Init code layout (after execution code)
    init_offset = dispatch_prefix_len + len(execution_code)
    init_loop_target = init_offset + 1 + len(calldata_load)  # +1 for JUMPDEST

    init_loop = (
        Op.JUMPDEST
        + calldata_load
        + Op.JUMPDEST
        + init_loop_body
        + Op.JUMPI(init_loop_target, loop_condition)
        + Op.STOP
    )

    # Combined contract: dispatch + execution + init
    contract_code = (
        Op.CALLDATASIZE
        + Op.PUSH1(64)
        + Op.EQ
        + Op.PUSH2(init_offset)
        + Op.JUMPI
        # Fall through to execution
        + execution_code
        + init_loop
    )

    # Deploy via EXTCODECOPY pattern
    contract_code_address = pre.deploy_contract(code=contract_code)
    creation_code = Op.EXTCODECOPY(
        address=contract_code_address,
        dest_offset=0,
        offset=0,
        size=Op.EXTCODESIZE(contract_code_address),
    ) + Op.RETURN(0, Op.MSIZE)

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

    # Create init transactions to initialize slots (uses 64-byte calldata)
    if init_slots_count > 0:
        slot_init_loop_cost = (
            gas_costs.G_STORAGE_SET
            + gas_costs.G_COLD_SLOAD
            + gas_costs.G_JUMPDEST
            + gas_costs.G_VERY_LOW
            * 12  # DUPs, SWAPs, PUSHs, SUB, ADD, ISZEROs
            + gas_costs.G_HIGH
        )

        max_slots_per_tx = (
            tx_gas_limit * 9 // 10 - intrinsic_gas_cost_calc(calldata=b"\xff" * 64)
        ) // slot_init_loop_cost

        for i in range(math.ceil(init_slots_count / max_slots_per_tx)):
            start_slot = 1 + i * max_slots_per_tx
            count = min(
                max_slots_per_tx, init_slots_count - i * max_slots_per_tx
            )
            calldata = start_slot.to_bytes(32, "big") + count.to_bytes(
                32, "big"
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
        # pytest.param(
        #     StorageAction.WRITE_SAME_VALUE,
        #     TransactionResult.OUT_OF_GAS,
        #     id="SSTORE same value, out of gas",
        # ),
        # pytest.param(
        #     StorageAction.WRITE_NEW_VALUE,
        #     TransactionResult.SUCCESS,
        #     id="SSTORE new value",
        # ),
        # pytest.param(
        #     StorageAction.WRITE_NEW_VALUE,
        #     TransactionResult.REVERT,
        #     id="SSTORE new value, revert",
        # ),
        # pytest.param(
        #     StorageAction.WRITE_NEW_VALUE,
        #     TransactionResult.OUT_OF_GAS,
        #     id="SSTORE new value, out of gas",
        # ),
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

    For forks with tx gas limit cap, execution is split into multiple
    transactions, each accessing different slot ranges to keep them cold.
    """
    gas_costs = fork.gas_costs()

    # Calculate loop cost based on storage action
    # Stack during loop: [count, start_slot]
    loop_cost = gas_costs.G_COLD_SLOAD  # All accesses are always cold
    if storage_action == StorageAction.WRITE_NEW_VALUE:
        loop_cost += (
            gas_costs.G_STORAGE_RESET
            if not absent_slots
            else gas_costs.G_STORAGE_SET
        )
    elif storage_action == StorageAction.WRITE_SAME_VALUE:
        loop_cost += (
            gas_costs.G_STORAGE_SET if absent_slots else gas_costs.G_WARM_SLOAD
        )

    # Build execution code body based on storage action
    # Stack: [count, start_slot] - use DUP2 to access start_slot
    execution_code_body = Bytecode()
    if storage_action == StorageAction.WRITE_SAME_VALUE:
        # DUP2 DUP1 SSTORE = SSTORE(slot, slot)
        execution_code_body = Op.DUP2 + Op.DUP1 + Op.SSTORE
        loop_cost += gas_costs.G_VERY_LOW * 2
    elif storage_action == StorageAction.WRITE_NEW_VALUE:
        # DUP2 NOT(0) SWAP1 SSTORE = SSTORE(slot, -1)
        execution_code_body = Op.DUP2 + Op.NOT(0) + Op.SWAP1 + Op.SSTORE
        loop_cost += gas_costs.G_VERY_LOW * 4
    elif storage_action == StorageAction.READ:
        # DUP2 SLOAD POP = SLOAD(slot), discard result
        execution_code_body = Op.POP(Op.SLOAD(Op.DUP2))
        loop_cost += gas_costs.G_VERY_LOW + gas_costs.G_BASE

    # Add loop overhead costs:
    # - JUMPDEST
    # - slot_increment: SWAP1 PUSH1(1) ADD SWAP1 (4 ops)
    # - loop_condition: PUSH1(1) SWAP1 SUB DUP1 ISZERO ISZERO (6 ops)
    # - PUSH2(target) for JUMPI
    # - JUMPI
    loop_cost += (
        gas_costs.G_JUMPDEST
        + gas_costs.G_VERY_LOW * 11  # 4 (increment) + 6 (condition) + 1 (PUSH2 target)
        + gas_costs.G_HIGH
    )

    # Prefix cost (dispatch + calldata loads, NOT including first JUMPDEST):
    # CALLDATASIZE PUSH1(64) EQ PUSH2(offset) JUMPI CALLDATALOAD(0) CALLDATALOAD(32)
    # Note: JUMPDEST is counted in loop_cost, not here
    prefix_cost = (
        gas_costs.G_BASE  # CALLDATASIZE
        + gas_costs.G_VERY_LOW * 3  # PUSH1(64) + EQ + PUSH2
        + gas_costs.G_HIGH  # JUMPI
        + gas_costs.G_VERY_LOW * 4  # 2x PUSH + 2x CALLDATALOAD
    )

    suffix_cost = (
        gas_costs.G_VERY_LOW * 2
        if tx_result == TransactionResult.REVERT
        else 0
    )

    # Helper to compute intrinsic cost for execution calldata
    # Note: Uses standard calldata cost (not EIP-7623 floor) to match EVM behavior
    def exec_intrinsic_for(calldata: bytes) -> int:
        calldata_cost = sum(4 if b == 0 else 16 for b in calldata)
        return 21000 + calldata_cost

    # Calculate max slots per execution tx (with 10% safety margin)
    # Use worst-case intrinsic estimate for safety margin calc
    max_exec_intrinsic = exec_intrinsic_for(b"\xff" * 65)
    max_slots_per_exec_tx = (
        tx_gas_limit * 9 // 10 - max_exec_intrinsic - prefix_cost
    ) // loop_cost

    # Estimate intrinsic for target slots calculation (start=1, count ~= large)
    est_calldata = (1).to_bytes(32, "big") + (10000).to_bytes(32, "big") + b"\x00"
    est_intrinsic = exec_intrinsic_for(est_calldata)
    num_target_slots = (
        gas_benchmark_value - est_intrinsic - prefix_cost - suffix_cost
    ) // loop_cost
    if tx_result == TransactionResult.OUT_OF_GAS:
        num_target_slots += 1

    # Setup: deploy contract and initialize slots
    contract_address, setup_txs = _setup_cold_storage_contract(
        pre=pre,
        fork=fork,
        init_slots_count=num_target_slots if not absent_slots else 0,
        execution_code_body=execution_code_body,
        tx_result=tx_result,
        tx_gas_limit=tx_gas_limit,
    )

    # Build execution transactions (may need multiple for high gas values)
    num_exec_txs = math.ceil(num_target_slots / max_slots_per_exec_tx)
    exec_txs: list[Transaction] = []
    total_gas_used = 0

    for i in range(num_exec_txs):
        start_slot = 1 + i * max_slots_per_exec_tx
        is_last_tx = i == num_exec_txs - 1

        if is_last_tx:
            slots_in_tx = num_target_slots - i * max_slots_per_exec_tx
        else:
            slots_in_tx = max_slots_per_exec_tx

        # Build calldata: (start_slot, count, padding)
        # Uses 65 bytes to distinguish from init's 64 bytes
        calldata = (
            start_slot.to_bytes(32, "big")
            + slots_in_tx.to_bytes(32, "big")
            + b"\x00"  # 1 byte padding
        )

        # Calculate gas for this transaction using actual calldata
        tx_intrinsic = exec_intrinsic_for(calldata)
        tx_gas = tx_intrinsic + prefix_cost + loop_cost * slots_in_tx
        if is_last_tx and tx_result == TransactionResult.REVERT:
            tx_gas += suffix_cost

        total_gas_used += tx_gas

        exec_txs.append(
            Transaction(
                to=contract_address,
                gas_limit=tx_gas,
                data=calldata,
                sender=pre.fund_eoa(),
            )
        )

    # Execution phase
    blocks = [Block(txs=setup_txs)]
    with TestPhaseManager.execution():
        blocks.append(Block(txs=exec_txs))

    # Post check: determine which slots have committed values
    post = {}
    if not absent_slots:
        # For REVERT/OUT_OF_GAS with multiple txs, only intermediate txs commit
        if num_exec_txs > 1 and tx_result in (
            TransactionResult.REVERT,
            TransactionResult.OUT_OF_GAS,
        ):
            committed_slots = (num_exec_txs - 1) * max_slots_per_exec_tx
        elif tx_result in (TransactionResult.REVERT, TransactionResult.OUT_OF_GAS):
            committed_slots = 0
        else:
            committed_slots = num_target_slots

        if committed_slots > 0:
            if (
                storage_action == StorageAction.WRITE_NEW_VALUE
                and tx_result == TransactionResult.SUCCESS
            ):
                storage = dict.fromkeys(range(1, committed_slots + 1), 2**256 - 1)
            else:
                storage = {i: i for i in range(1, committed_slots + 1)}
            post = {contract_address: Account(storage=storage)}

    benchmark_test(
        blocks=blocks,
        expected_benchmark_gas_used=(
            total_gas_used
            if tx_result != TransactionResult.OUT_OF_GAS
            else gas_benchmark_value
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
