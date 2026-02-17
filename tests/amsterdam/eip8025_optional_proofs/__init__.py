"""Tests for EIP-8025 optional proofs witness coverage."""

# Witness coverage matrix for client-triage.
#
# Node coverage:
# - test_storage_trie_insert_then_delete_sibling (test_tree_sibiling.py):
#   Branch compression over sibling set with insert/delete ordering.
# - test_runtime_only_branch_compression_not_in_witness
#   (test_tree_sibiling.py):
#   Runtime-only storage nodes must not leak into pre-state witness nodes.
# - test_witness_branch_compression_delete_then_insert_sibling
#   (test_witness_nodes.py):
#   Reverse write ordering over sibling branches.
# - test_witness_branch_compression_update_then_delete_multiple_siblings
#   (test_witness_nodes.py):
#   Multi-sibling collapse can over-capture survivor siblings.
# - test_witness_nodes_dirty_slots_without_reads (test_witness_nodes.py):
#   Dirty-key-only paths still must capture pre-state trie nodes.
# - test_witness_nodes_delete_all_prestate_slots (test_witness_nodes.py):
#   Full pre-state delete should still include touched pre-state paths.
# - test_witness_nodes_runtime_only_create_and_delete_to_empty
#   (test_witness_nodes.py):
#   Runtime-created-and-deleted nodes must not be included.
# - test_witness_nodes_missing_slot_read_only (test_witness_nodes.py):
#   Missing-key reads on non-empty pre-state must capture absence paths.
# - test_witness_nodes_transient_new_slot_on_non_empty_prestate
#   (test_witness_nodes.py):
#   Runtime-only insert/delete under non-empty pre-state must avoid leaks.
#
# Bytecode coverage:
# - test_witness_bytecodes_call_family (test_witness_bytecodes.py):
#   CALL/CALLCODE/DELEGATECALL/STATICCALL bytecode tracking.
# - test_witness_bytecodes_extcode_ops_and_dedup
#   (test_witness_bytecodes.py):
#   EXTCODE tracking and hash-based dedup behavior.
# - test_witness_bytecodes_excludes_runtime_created_code
#   (test_witness_bytecodes.py):
#   In-block created code must not appear in witness bytecodes.
# - test_witness_bytecodes_empty_eoa_is_not_tracked
#   (test_witness_bytecodes.py):
#   Empty EOA code must not be emitted as witness bytecode.
# - test_witness_bytecodes_7702_sender_code_tracked
#   (test_witness_bytecodes.py):
#   Type-4 sender pre-state code path must be tracked.
# - test_witness_bytecodes_7702_authority_preexisting_code_tracked
#   (test_witness_bytecodes.py):
#   Type-4 authority pre-existing code path must be tracked.
# - test_witness_bytecodes_extcodehash_only_does_not_track_code
#   (test_witness_bytecodes.py):
#   EXTCODEHASH-only path should not force bytecode witness inclusion.
# - test_witness_bytecodes_7702_authority_new_code_is_not_tracked
#   (test_witness_bytecodes.py):
#   Type-4 runtime-written authority code must not be treated as pre-state.
#
# Ancestor coverage:
# - test_witness_ancestors_parent_only_baseline
#   (test_witness_ancestors.py):
#   Parent/system baseline ancestor inclusion.
# - test_witness_ancestors_oldest_blockhash_extends_range
#   (test_witness_ancestors.py):
#   Oldest accessed BLOCKHASH must anchor ancestor range.
# - test_witness_ancestors_current_and_future_blockhash_do_not_extend_range
#   (test_witness_ancestors.py):
#   Current/future/out-of-range accesses must not extend ancestor range.
# - test_witness_ancestors_oldest_of_multiple_accesses
#   (test_witness_ancestors.py):
#   Multiple BLOCKHASH accesses must choose the oldest anchor.
# - test_witness_ancestors_blockhash_window_boundary
#   (test_witness_ancestors.py):
#   Exact 256-block boundary: N-256 valid, N-257 invalid.
#
# Failure-path coverage (OOG / stack overflow):
# - test_witness_oog_before_call_access (test_witness_failure_paths.py):
#   OOG before access must not capture target account/code.
# - test_witness_oog_before_call_access_plus_one_gas
#   (test_witness_failure_paths.py):
#   +1 gas at the same boundary should include target access in witness.
# - test_witness_oog_before_delegation_access
#   (test_witness_failure_paths.py):
#   OOG before delegation lookup must not capture pointer/delegated code.
# - test_witness_oog_before_delegation_access_plus_one_gas
#   (test_witness_failure_paths.py):
#   +1 gas at the same boundary should include delegation-path access.
# - test_witness_subcall_oog_after_access (test_witness_failure_paths.py):
#   Access done first, later OOG should still keep captured access.
# - test_witness_call_insufficient_balance_after_access
#   (test_witness_failure_paths.py):
#   CALL failure after access checks must still preserve accessed targets.
# - test_witness_revert_before_state_access
#   (test_witness_failure_paths.py):
#   REVERT before opcode execution must not capture unreachable accesses.
# - test_witness_revert_after_state_access
#   (test_witness_failure_paths.py):
#   REVERT after access must preserve already-captured witness data.
# - test_witness_stack_overflow_before_state_access
#   (test_witness_failure_paths.py):
#   Overflow before opcode execution must not capture that access.
# - test_witness_stack_overflow_after_state_access
#   (test_witness_failure_paths.py):
#   Overflow after access must preserve already-captured access.
#
# Osaka fix linkage:
# - Wrong node capture ordering for dirty keys/accounts: b48e55e82
# - Runtime-created/edited trie nodes leaking into witness: b95429a83
# - Runtime-created/changed bytecode included in witness: 1f2c6c67e
# - Missing 7702 sender/authority bytecode tracking: d5f8a5e6e, 50adb168d
# - 7702 gas/access sequencing edge behavior: f290ecba9
# - Ancestor range generation: 2331b719c
