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

    loop_cost = gas_costs.G_COLD_SLOAD  # All accesses are always cold
    if storage_action == StorageAction.WRITE_NEW_VALUE:
        if not absent_slots:
            loop_cost += gas_costs.G_STORAGE_RESET
        else:
            loop_cost += gas_costs.G_STORAGE_SET
    elif storage_action == StorageAction.WRITE_SAME_VALUE:
        if absent_slots:
            loop_cost += gas_costs.G_STORAGE_SET
        else:
            loop_cost += gas_costs.G_WARM_SLOAD
    elif storage_action == StorageAction.READ:
        loop_cost += 0  # Only G_COLD_SLOAD is charged

    # Contract code
    execution_code_body = Bytecode()
    if storage_action == StorageAction.WRITE_SAME_VALUE:
        # All the storage slots in the contract are initialized to their index.
        # That is, storage slot `i` is initialized to `i`.
        execution_code_body = Op.SSTORE(Op.DUP1, Op.DUP1)
        loop_cost += gas_costs.G_VERY_LOW * 2
    elif storage_action == StorageAction.WRITE_NEW_VALUE:
        # The new value 2^256-1 is guaranteed to be different from the initial
        # value.
        execution_code_body = Op.SSTORE(Op.DUP2, Op.NOT(0))
        loop_cost += gas_costs.G_VERY_LOW * 3
    elif storage_action == StorageAction.READ:
        execution_code_body = Op.POP(Op.SLOAD(Op.DUP1))
        loop_cost += gas_costs.G_VERY_LOW + gas_costs.G_BASE

    # Add costs jump-logic costs
    loop_cost += (
        gas_costs.G_JUMPDEST  # Prefix Jumpdest
        + gas_costs.G_VERY_LOW * 7  # ISZEROs, PUSHs, SWAPs, SUB, DUP
        + gas_costs.G_HIGH  # JUMPI
    )

    prefix_cost = (
        gas_costs.G_VERY_LOW  # Target slots push
    )
    if not absent_slots:
        # Additional cost for init check prefix in execution phase
        # CALLDATASIZE + ISZERO + PUSH2 + JUMPI + JUMPDEST
        prefix_cost += (
            gas_costs.G_BASE  # CALLDATASIZE
            + gas_costs.G_VERY_LOW  # ISZERO
            + gas_costs.G_VERY_LOW  # PUSH2
            + gas_costs.G_HIGH  # JUMPI
            + gas_costs.G_JUMPDEST  # outer JUMPDEST
        )

    suffix_cost = 0
    if tx_result == TransactionResult.REVERT:
        suffix_cost = (
            gas_costs.G_VERY_LOW * 2  # Revert PUSHs
        )

    num_target_slots = (
        gas_benchmark_value
        - intrinsic_gas_cost_calc()
        - prefix_cost
        - suffix_cost
    ) // loop_cost
    if tx_result == TransactionResult.OUT_OF_GAS:
        # Add an extra slot to make it run out-of-gas
        num_target_slots += 1

    code_prefix = Op.PUSH4(num_target_slots) + Op.JUMPDEST
    code_loop = execution_code_body + Op.JUMPI(
        len(code_prefix) - 1,
        Op.PUSH1(1) + Op.SWAP1 + Op.SUB + Op.DUP1 + Op.ISZERO + Op.ISZERO,
    )
    execution_code = code_prefix + code_loop

    if tx_result == TransactionResult.REVERT:
        execution_code += Op.REVERT(0, 0)
    else:
        execution_code += Op.STOP

    execution_code_address = pre.deploy_contract(code=execution_code)

    total_gas_used = (
        num_target_slots * loop_cost
        + intrinsic_gas_cost_calc()
        + prefix_cost
        + suffix_cost
    )

    # Contract creation
    sender_addr = pre.fund_eoa()
    setup_txs = []

    if absent_slots:
        # No slot initialization needed - simple case
        creation_code = (
            Op.EXTCODECOPY(
                address=execution_code_address,
                dest_offset=0,
                offset=0,
                size=Op.EXTCODESIZE(execution_code_address),
            )
            + Op.RETURN(0, Op.MSIZE)
        )
        with TestPhaseManager.setup():
            setup_txs.append(
                Transaction(
                    to=None,
                    gas_limit=tx_gas_limit,
                    data=creation_code,
                    sender=sender_addr,
                )
            )
    else:
        # Deploy contract with init entry point, then batch initialize slots.
        # Contract code has two paths:
        # - If calldata present: run init loop from (start_slot, count)
        # - If no calldata: run execution code

        # Gas cost per slot initialization (for batch size calculation)
        slot_init_loop_cost = (
            gas_costs.G_STORAGE_SET  # 20,000 - storing to empty slot
            + gas_costs.G_COLD_SLOAD  # 2,100 - cold access
            + gas_costs.G_JUMPDEST  # 1
            + gas_costs.G_VERY_LOW * 3  # DUPs
            + gas_costs.G_VERY_LOW * 3  # SWAPs
            + gas_costs.G_VERY_LOW * 2  # PUSHs
            + gas_costs.G_VERY_LOW  # SUB
            + gas_costs.G_VERY_LOW  # ADD
            + gas_costs.G_VERY_LOW * 2  # ISZEROs
            + gas_costs.G_HIGH  # JUMPI
        )

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
                condition=Op.PUSH1(1)
                + Op.SWAP1
                + Op.SUB
                + Op.DUP1
                + Op.ISZERO
                + Op.ISZERO,
            )
            + Op.STOP
        )

        # Combined contract: init check + init code + execution code
        # Jump to execution if no calldata
        # Prefix: CALLDATASIZE + ISZERO + PUSH2 + JUMPI = 6 bytes
        prefix_len = 6
        execution_offset = prefix_len + len(init_loop)

        # Rebuild execution_code with adjusted jump target for embedding
        # The loop's JUMPDEST will be at: prefix + init_loop + JUMPDEST + code_prefix
        execution_code_start = prefix_len + len(init_loop) + 1  # +1 for outer JUMPDEST
        adjusted_code_prefix = Op.PUSH4(num_target_slots) + Op.JUMPDEST
        adjusted_code_loop = execution_code_body + Op.JUMPI(
            execution_code_start + len(adjusted_code_prefix) - 1,
            Op.PUSH1(1) + Op.SWAP1 + Op.SUB + Op.DUP1 + Op.ISZERO + Op.ISZERO,
        )
        adjusted_execution_code = adjusted_code_prefix + adjusted_code_loop

        if tx_result == TransactionResult.REVERT:
            adjusted_execution_code += Op.REVERT(0, 0)
        else:
            adjusted_execution_code += Op.STOP

        contract_code = (
            Op.CALLDATASIZE
            + Op.ISZERO
            + Op.PUSH2(execution_offset)
            + Op.JUMPI
            + init_loop
            + Op.JUMPDEST  # execution_start
            + adjusted_execution_code
        )

        # Deploy contract_code to a helper address, then use EXTCODECOPY
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

        with TestPhaseManager.setup():
            setup_txs.append(
                Transaction(
                    to=None,
                    gas_limit=tx_gas_limit,
                    data=creation_code,
                    sender=sender_addr,
                )
            )

        # Calculate contract address before creating init transactions
        contract_address = compute_create_address(
            address=sender_addr, nonce=0
        )

        # Calculate batch sizes and create init transactions
        # Use 90% of gas limit to leave margin for overhead
        max_slots_per_tx = (
            math.floor(tx_gas_limit * 0.9) - intrinsic_gas_cost_calc()
        ) // slot_init_loop_cost
        num_init_txs = math.ceil(num_target_slots / max_slots_per_tx)

        print("Num init txs:", num_init_txs)
        print("Max slots per tx:", max_slots_per_tx)

        for i in range(num_init_txs):
            # Slots 1 to num_target_slots (matching execution loop which decrements from n to 1)
            start_slot = 1 + i * max_slots_per_tx
            count = min(max_slots_per_tx, num_target_slots - i * max_slots_per_tx)

            # Calldata: (start_slot, count) as 32-byte words
            calldata = (
                start_slot.to_bytes(32, "big") + count.to_bytes(32, "big")
            )

            setup_txs.append(
                Transaction(
                    to=contract_address,
                    gas_limit=tx_gas_limit,
                    data=calldata,
                    sender=pre.fund_eoa(),
                )
            )

    blocks = [Block(txs=setup_txs)]

    contract_address = compute_create_address(address=sender_addr, nonce=0)

    with TestPhaseManager.execution():
        op_tx = Transaction(
            to=contract_address,
            gas_limit=gas_benchmark_value,
            sender=pre.fund_eoa(),
        )
    blocks.append(Block(txs=[op_tx]))

    # Post check: verify storage slots were initialized correctly
    post = {}
    if not absent_slots:
        # Slots 1 to num_target_slots should be initialized
        # Value depends on storage_action and tx_result
        if (
            storage_action == StorageAction.WRITE_NEW_VALUE
            and tx_result == TransactionResult.SUCCESS
        ):
            # Execution wrote NOT(0) to all slots
            expected_value = 2**256 - 1
        else:
            # Slots retain their initialized value (index)
            # (READ doesn't change, WRITE_SAME_VALUE writes same, REVERT/OOG reverts)
            expected_value = None  # Each slot has its index as value

        if expected_value is not None:
            storage = {i: expected_value for i in range(1, num_target_slots + 1)}
        else:
            storage = {i: i for i in range(1, num_target_slots + 1)}

        post = {contract_address: Account(storage=storage)}

    benchmark_test(
        blocks=blocks,
        # skip_gas_used_validation=True,
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
