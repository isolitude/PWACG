# v5 Changelog

## Added Anti-pattern 6: Sbc-first argument order

ALL `calculate_*` and `lasso_calculate_*` methods MUST use `(sbc_phi, sbc_f, amp)` argument order, NOT `(amp, sbc_phi, sbc_f)`.

Root cause: LLM generated amp-first signatures for pull, causing the 3D amp tensor `(damp, n_events, 2)` to be passed where a 1D `(n_events,)` Sbc was expected, triggering einsum subscript shape mismatches.

Fix: 6 function signatures + 16 callsites in pull_object_kk.py changed to Sbc-first order.
