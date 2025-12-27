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


# =============================================================================
# Cold Storage Benchmark - Helper Functions
# =============================================================================

# Calldata sizes for mode dispatch:
# - Init mode:  64 bytes (start_slot + count)
# - Exec mode:  65 bytes (start_slot + count + 1 padding byte)
INIT_CALLDATA_SIZE = 64
EXEC_CALLDATA_SIZE = 65


def _calc_intrinsic_gas(calldata: bytes) -> int:
    """
    Calculate transaction intrinsic gas for calldata.

    Uses standard calldata cost (4 gas/zero byte, 16/non-zero).
    Does NOT apply EIP-7623 floor - we need exact gas matching.
    """
    calldata_cost = sum(4 if b == 0 else 16 for b in calldata)
    return 21000 + calldata_cost


def _build_cold_storage_contract(
    execution_code_body: Bytecode,
    ends_with_revert: bool,
) -> Bytecode:
    """
    Build contract bytecode for cold storage benchmarking.

    Contract has two modes controlled by calldata size:
    - CALLDATASIZE == 64: Init mode - SSTORE(slot, slot) for each slot
    - CALLDATASIZE != 64: Exec mode - run execution_code_body for each slot

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
    # Dispatch: CALLDATASIZE PUSH1(64) EQ PUSH2(init_offset) JUMPI = 8 bytes
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
            calldata = start.to_bytes(32, "big") + count.to_bytes(32, "big")

            setup_txs.append(
                Transaction(
                    to=contract_address,
                    gas_limit=tx_gas_limit,
                    data=calldata,
                    sender=pre.fund_eoa(),
                )
            )

    return contract_address, setup_txs


# =============================================================================
# Cold Storage Benchmark - Main Test
# =============================================================================


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

    Test matrix:
    - storage_action: READ (SLOAD), WRITE_SAME_VALUE, WRITE_NEW_VALUE
    - absent_slots: True = slots never initialized, False = pre-initialized
    - tx_result: SUCCESS, REVERT (all undone), OUT_OF_GAS (partial)
    """
    gas_costs = fork.gas_costs()

    # -------------------------------------------------------------------------
    # Step 1: Calculate gas costs per loop iteration
    # -------------------------------------------------------------------------

    # All storage accesses are cold (2100 gas)
    storage_access_cost = gas_costs.G_COLD_SLOAD

    # Additional cost depends on storage action
    if storage_action == StorageAction.READ:
        # SLOAD only - no additional storage cost
        storage_op_cost = 0
        # Bytecode: DUP2 SLOAD POP (access slot, discard result)
        execution_code_body = Op.POP(Op.SLOAD(Op.DUP2))
        bytecode_cost = gas_costs.G_VERY_LOW + gas_costs.G_BASE

    elif storage_action == StorageAction.WRITE_SAME_VALUE:
        # SSTORE same value: cold read + warm write (or cold if absent)
        if absent_slots:
            storage_op_cost = gas_costs.G_STORAGE_SET
        else:
            storage_op_cost = gas_costs.G_WARM_SLOAD
        # Bytecode: DUP2 DUP1 SSTORE (write slot number to slot)
        execution_code_body = Op.DUP2 + Op.DUP1 + Op.SSTORE
        bytecode_cost = gas_costs.G_VERY_LOW * 2

    elif storage_action == StorageAction.WRITE_NEW_VALUE:
        # SSTORE new value: cold read + storage modification
        if absent_slots:
            storage_op_cost = gas_costs.G_STORAGE_SET
        else:
            storage_op_cost = gas_costs.G_STORAGE_RESET
        # Bytecode: DUP2 NOT(0) SWAP1 SSTORE (write -1 to slot)
        execution_code_body = Op.DUP2 + Op.NOT(0) + Op.SWAP1 + Op.SSTORE
        bytecode_cost = gas_costs.G_VERY_LOW * 4

    # Loop overhead per iteration:
    # - JUMPDEST (1)
    # - slot_increment: SWAP1 PUSH1 ADD SWAP1 (4 ops)
    # - loop_condition: PUSH1 SWAP1 SUB DUP1 ISZERO ISZERO (6 ops)
    # - PUSH2 for jump target (1 op)
    # - JUMPI (1)
    loop_overhead = (
        gas_costs.G_JUMPDEST
        + gas_costs.G_VERY_LOW * 11  # 4 + 6 + 1 = 11 G_VERY_LOW ops
        + gas_costs.G_HIGH  # JUMPI
    )

    loop_cost = (
        storage_access_cost + storage_op_cost + bytecode_cost + loop_overhead
    )

    # -------------------------------------------------------------------------
    # Step 2: Calculate prefix and suffix costs (one-time per transaction)
    # -------------------------------------------------------------------------

    # Dispatch + calldata load (before loop starts):
    # CALLDATASIZE PUSH1 EQ PUSH2 JUMPI CALLDATALOAD(0) CALLDATALOAD(32)
    prefix_cost = (
        gas_costs.G_BASE  # CALLDATASIZE
        + gas_costs.G_VERY_LOW * 3  # PUSH1 + EQ + PUSH2
        + gas_costs.G_HIGH  # JUMPI
        + gas_costs.G_VERY_LOW * 4  # 2x (PUSH1 + CALLDATALOAD)
    )

    # Suffix cost (REVERT only): PUSH1(0) PUSH1(0) REVERT
    if tx_result == TransactionResult.REVERT:
        suffix_cost = gas_costs.G_VERY_LOW * 2
    else:
        suffix_cost = 0

    # -------------------------------------------------------------------------
    # Step 3: Calculate number of slots to access
    # -------------------------------------------------------------------------

    # Estimate intrinsic for a typical execution calldata
    typical_calldata = (
        (1).to_bytes(32, "big") + (10000).to_bytes(32, "big") + b"\x00"
    )
    typical_intrinsic = _calc_intrinsic_gas(typical_calldata)

    # How many slots can we access with gas_benchmark_value?
    num_target_slots = (
        gas_benchmark_value - typical_intrinsic - prefix_cost - suffix_cost
    ) // loop_cost

    # For OUT_OF_GAS, we need one extra slot to trigger OOG
    if tx_result == TransactionResult.OUT_OF_GAS:
        num_target_slots += 1

    # -------------------------------------------------------------------------
    # Step 4: Deploy contract and initialize storage
    # -------------------------------------------------------------------------

    contract_address, setup_txs = _deploy_cold_storage_contract(
        pre=pre,
        fork=fork,
        execution_code_body=execution_code_body,
        ends_with_revert=(tx_result == TransactionResult.REVERT),
        init_slot_count=num_target_slots if not absent_slots else 0,
        tx_gas_limit=tx_gas_limit,
    )

    # -------------------------------------------------------------------------
    # Step 5: Build execution transactions
    # -------------------------------------------------------------------------

    # Calculate max slots per tx (with 10% safety margin)
    worst_intrinsic = _calc_intrinsic_gas(b"\xff" * EXEC_CALLDATA_SIZE)
    available_gas = tx_gas_limit * 9 // 10 - worst_intrinsic - prefix_cost
    max_slots_per_tx = available_gas // loop_cost

    num_exec_txs = math.ceil(num_target_slots / max_slots_per_tx)
    exec_txs: list[Transaction] = []
    total_gas_used = 0

    for tx_index in range(num_exec_txs):
        start_slot = 1 + tx_index * max_slots_per_tx
        is_last_tx = tx_index == num_exec_txs - 1

        # Last tx gets remaining slots; others get max_slots_per_tx
        if is_last_tx:
            slots_in_tx = num_target_slots - tx_index * max_slots_per_tx
        else:
            slots_in_tx = max_slots_per_tx

        # Calldata: (start_slot, count, padding_byte)
        calldata = (
            start_slot.to_bytes(32, "big")
            + slots_in_tx.to_bytes(32, "big")
            + b"\x00"
        )
        tx_intrinsic = _calc_intrinsic_gas(calldata)

        # Calculate gas limit for this transaction
        # For OOG: give gas for (slots - 1) to trigger OOG on last iteration
        if is_last_tx and tx_result == TransactionResult.OUT_OF_GAS:
            slots_for_gas = slots_in_tx - 1
        else:
            slots_for_gas = slots_in_tx

        tx_gas = tx_intrinsic + prefix_cost + loop_cost * slots_for_gas
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

    # -------------------------------------------------------------------------
    # Step 6: Build expected post-state
    # -------------------------------------------------------------------------

    # Determine expected storage values after execution.
    # Key insight: init phase writes slot[i] = i for all slots.
    # These values persist unless overwritten by a committed transaction.
    post = {}
    if not absent_slots:
        # How many slots have "execution" values (vs init values)?
        if tx_result == TransactionResult.SUCCESS:
            # All slots modified by execution
            slots_with_exec_values = num_target_slots
        elif tx_result == TransactionResult.REVERT:
            # REVERT is in contract code, so ALL txs revert
            slots_with_exec_values = 0
        elif num_exec_txs > 1:
            # Multi-tx OOG: intermediate txs complete, last tx reverts
            slots_with_exec_values = (num_exec_txs - 1) * max_slots_per_tx
        else:
            # Single-tx OOG: entire tx reverts
            slots_with_exec_values = 0

        # Build storage expectations
        storage = {}
        if storage_action == StorageAction.WRITE_NEW_VALUE:
            # Slots modified by execution have value -1 (0xff...ff)
            for i in range(1, slots_with_exec_values + 1):
                storage[i] = 2**256 - 1
            # Remaining slots keep init value (slot[i] = i)
            for i in range(slots_with_exec_values + 1, num_target_slots + 1):
                storage[i] = i
        else:
            # READ or WRITE_SAME_VALUE: all slots have init value
            for i in range(1, num_target_slots + 1):
                storage[i] = i

        if storage:
            post = {contract_address: Account(storage=storage)}

    # -------------------------------------------------------------------------
    # Step 7: Execute benchmark
    # -------------------------------------------------------------------------

    blocks = [Block(txs=setup_txs)]
    with TestPhaseManager.execution():
        blocks.append(Block(txs=exec_txs))

    benchmark_test(
        blocks=blocks,
        expected_benchmark_gas_used=total_gas_used,
        post=post,
    )


# =============================================================================
# Warm Storage Benchmark
# =============================================================================


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
