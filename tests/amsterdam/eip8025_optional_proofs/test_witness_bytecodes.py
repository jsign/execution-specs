"""
Witness-focused bytecode coverage for EIP-8025 optional proofs.

These cases stress bytecode inclusion/exclusion rules in the Osaka deferred
witness path.
"""

import pytest
from execution_testing import (
    Account,
    Alloc,
    AuthorizationTuple,
    Block,
    BlockchainTestFiller,
    Initcode,
    Op,
    Storage,
    Transaction,
)

pytestmark = pytest.mark.valid_at("Osaka")

REFERENCE_SPEC_GIT_PATH = "TODO"
REFERENCE_SPEC_VERSION = "TODO"


CALL_FAMILY_BUILDERS = [
    pytest.param(
        lambda address: Op.CALL(gas=100_000, address=address),
        id="call",
    ),
    pytest.param(
        lambda address: Op.CALLCODE(gas=100_000, address=address),
        id="callcode",
    ),
    pytest.param(
        lambda address: Op.DELEGATECALL(gas=100_000, address=address),
        id="delegatecall",
    ),
    pytest.param(
        lambda address: Op.STATICCALL(gas=100_000, address=address),
        id="staticcall",
    ),
]


def _delegation_designation(address: bytes) -> bytes:
    """Return EIP-7702 delegation designation bytecode for `address`."""
    return b"\xef\x01\x00" + address


@pytest.mark.parametrize("call_builder", CALL_FAMILY_BUILDERS)
def test_witness_bytecodes_call_family(
    pre: Alloc,
    blockchain_test: BlockchainTestFiller,
    call_builder,
) -> None:
    """
    Access target bytecode through each call-family opcode variant.

    This ensures all call paths feed bytecode witness collection.
    """
    sender = pre.fund_eoa()
    callee = pre.deploy_contract(Op.STOP)
    caller = pre.deploy_contract(
        code=Op.SSTORE(0, call_builder(callee)) + Op.STOP,
        storage={},
    )

    tx = Transaction(sender=sender, to=caller, gas_limit=300_000)

    blockchain_test(
        pre=pre,
        blocks=[Block(txs=[tx])],
        post={caller: Account(storage={0: 1})},
    )


def test_witness_bytecodes_extcode_ops_and_dedup(
    pre: Alloc,
    blockchain_test: BlockchainTestFiller,
) -> None:
    """
    Access the same bytecode via two addresses and multiple opcodes.

    The witness bytecode set should be deduplicated by code hash.
    """
    sender = pre.fund_eoa()

    shared_code = Op.PUSH1(0x42) + Op.STOP
    shared_code_size = len(shared_code)

    contract_a = pre.deploy_contract(shared_code)
    contract_b = pre.deploy_contract(shared_code)

    probe = pre.deploy_contract(
        code=(
            Op.EXTCODECOPY(address=contract_a, size=1)
            + Op.SSTORE(0, Op.CALL(gas=100_000, address=contract_a))
            + Op.SSTORE(1, Op.CALL(gas=100_000, address=contract_b))
            + Op.SSTORE(
                2,
                Op.ADD(
                    Op.EXTCODESIZE(contract_a),
                    Op.EXTCODESIZE(contract_b),
                ),
            )
            + Op.STOP
        ),
        storage={},
    )

    tx = Transaction(sender=sender, to=probe, gas_limit=500_000)

    blockchain_test(
        pre=pre,
        blocks=[Block(txs=[tx])],
        post={
            probe: Account(
                storage={
                    0: 1,
                    1: 1,
                    2: 2 * shared_code_size,
                }
            )
        },
    )


def test_witness_bytecodes_excludes_runtime_created_code(
    pre: Alloc,
    blockchain_test: BlockchainTestFiller,
) -> None:
    """
    Create a contract and access it in the same transaction.

    Runtime-created code is valid for execution but must not be included in
    pre-state witness bytecodes.
    """
    sender = pre.fund_eoa()

    runtime_code = Op.SSTORE(0xCAFE, 1) + Op.STOP
    initcode = Initcode(deploy_code=runtime_code)

    slot_created = 0
    slot_create_failed = 1
    slot_created_codesize = 2
    slot_call_result = 3

    expected_storage = Storage(
        {
            slot_create_failed: 0,
            slot_created_codesize: len(runtime_code),
            slot_call_result: 1,
        }
    )
    expected_storage.set_expect_any(slot_created)

    creator = pre.deploy_contract(
        code=(
            Op.CALLDATACOPY(0, 0, Op.CALLDATASIZE)
            + Op.SSTORE(
                slot_created,
                Op.CREATE(value=0, offset=0, size=Op.CALLDATASIZE),
            )
            + Op.SSTORE(
                slot_create_failed,
                Op.ISZERO(Op.SLOAD(slot_created)),
            )
            + Op.SSTORE(
                slot_created_codesize,
                Op.EXTCODESIZE(Op.SLOAD(slot_created)),
            )
            + Op.SSTORE(
                slot_call_result,
                Op.CALL(gas=100_000, address=Op.SLOAD(slot_created)),
            )
            + Op.STOP
        ),
        storage={},
    )

    tx = Transaction(
        sender=sender,
        to=creator,
        data=initcode,
        gas_limit=1_000_000,
    )

    blockchain_test(
        pre=pre,
        blocks=[Block(txs=[tx])],
        post={creator: Account(storage=expected_storage)},
    )


def test_witness_bytecodes_empty_eoa_is_not_tracked(
    pre: Alloc,
    blockchain_test: BlockchainTestFiller,
) -> None:
    """
    Access an EOA with empty code via EXTCODESIZE and CALL.

    Empty code accesses should not create witness bytecode entries.
    """
    sender = pre.fund_eoa()
    eoa_target = pre.fund_eoa()

    probe = pre.deploy_contract(
        code=(
            Op.SSTORE(0, Op.EXTCODESIZE(eoa_target))
            + Op.SSTORE(1, Op.CALL(gas=100_000, address=eoa_target))
            + Op.STOP
        ),
        storage={},
    )

    tx = Transaction(sender=sender, to=probe, gas_limit=300_000)

    blockchain_test(
        pre=pre,
        blocks=[Block(txs=[tx])],
        post={probe: Account(storage={0: 0, 1: 1})},
    )


def test_witness_bytecodes_extcodehash_only_does_not_track_code(
    pre: Alloc,
    blockchain_test: BlockchainTestFiller,
) -> None:
    """
    Access code hash only via EXTCODEHASH.

    This should not require bytecode witness entries for the target account.
    """
    sender = pre.fund_eoa()
    target = pre.deploy_contract(Op.PUSH1(0x01) + Op.STOP)

    probe = pre.deploy_contract(
        code=Op.SSTORE(0, Op.ISZERO(Op.EXTCODEHASH(target))) + Op.STOP,
        storage={},
    )

    tx = Transaction(sender=sender, to=probe, gas_limit=250_000)

    blockchain_test(
        pre=pre,
        blocks=[Block(txs=[tx])],
        post={probe: Account(storage={0: 0})},
    )


def test_witness_bytecodes_7702_sender_code_tracked(
    pre: Alloc,
    blockchain_test: BlockchainTestFiller,
) -> None:
    """
    Sender already has delegation designation code in pre-state.

    This targets sender-code bytecode tracking in transaction validation.
    """
    delegated_impl = pre.deploy_contract(Op.STOP)
    delegated_sender = pre.fund_eoa(delegation=delegated_impl)
    recipient = pre.deploy_contract(Op.SSTORE(0, 1) + Op.STOP)

    tx = Transaction(
        sender=delegated_sender,
        to=recipient,
        gas_limit=250_000,
    )

    blockchain_test(
        pre=pre,
        blocks=[Block(txs=[tx])],
        post={recipient: Account(storage={0: 1})},
    )


def test_witness_bytecodes_7702_authority_preexisting_code_tracked(
    pre: Alloc,
    blockchain_test: BlockchainTestFiller,
) -> None:
    """
    Authorization signer has pre-existing delegation code.

    This targets authority-code tracking during authorization processing.
    """
    sender = pre.fund_eoa()
    old_impl = pre.deploy_contract(Op.STOP)
    new_impl = pre.deploy_contract(Op.STOP)
    authority = pre.fund_eoa(delegation=old_impl)
    authority_start_nonce = int(authority.nonce)  # type: ignore[attr-defined]

    tx = Transaction(
        sender=sender,
        to=authority,
        gas_limit=500_000,
        authorization_list=[
            AuthorizationTuple(
                address=new_impl,
                nonce=authority_start_nonce,
                signer=authority,
            )
        ],
    )

    blockchain_test(
        pre=pre,
        blocks=[Block(txs=[tx])],
        post={
            authority: Account(
                nonce=authority_start_nonce + 1,
                code=_delegation_designation(new_impl),
            )
        },
    )


def test_witness_bytecodes_7702_authority_new_code_is_not_tracked(
    pre: Alloc,
    blockchain_test: BlockchainTestFiller,
) -> None:
    """
    Authorization signer starts with empty pre-state code.

    Delegation designation code written during authorization processing is
    runtime-modified account code and must not be tracked as pre-state
    witness bytecode.
    """
    sender = pre.fund_eoa()
    delegated_impl = pre.deploy_contract(Op.STOP)
    authority = pre.fund_eoa()
    authority_start_nonce = int(authority.nonce)  # type: ignore[attr-defined]

    tx = Transaction(
        sender=sender,
        to=authority,
        gas_limit=500_000,
        authorization_list=[
            AuthorizationTuple(
                address=delegated_impl,
                nonce=authority_start_nonce,
                signer=authority,
            )
        ],
    )

    blockchain_test(
        pre=pre,
        blocks=[Block(txs=[tx])],
        post={
            authority: Account(
                nonce=authority_start_nonce + 1,
                code=_delegation_designation(delegated_impl),
            )
        },
    )
