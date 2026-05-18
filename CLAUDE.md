# CLAUDE.md — PWACG Project Guide

## Project Overview

PWACG (Partial Wave Analysis Code Generator) generates JAX-accelerated physics fitting code for resonance analysis in high-energy physics (J/ψ → γ K⁺K⁻ decay). It replaces the old Jinja2 template system with LLM-based code generation.

## Architecture (5 layers)

| Layer | Path | Role |
|-------|------|------|
| L0 Schema | `create_code/schema/` | Pydantic v2 models for 3 config files (pwa_info, generator, parameters) |
| L1 IR | `create_code/ir/` | Rule-based builder: config → `PWAIR` (structured intermediate representation, no LLM) |
| L2 Codegen | `create_code/codegen/` | IR → LLM call → Python file; cache lookup, golden fallback |
| L3 Validate | `create_code/validate/` | syntax (ast) → types (pyright) → structural (methods check) |
| L4 Cache | `create_code/cache/` | SHA256-keyed content store + golden symlink fallback |

## Directory Structure

```
PWACG/
├── config/              # pwa_info_kk.json, generator_kk.json, parameters.json, latex.json
├── create_code/         # All generation pipeline code (IR builder, LLM client, cache, validate)
│   └── codegen/prompts/system.md  # LLM system prompt with patterns and S6a rules
├── rendered_scripts/    # Generated fit/batch/lasso/pull/select/draw_* modules
├── run/                 # Generated entry-point scripts
├── dlib/dplex.py        # Complex tensor library: deinsum, dconstruct, dabs, ddivide
├── Tensor/              # Tensor contraction utilities (BASE.py, RunCacheTensor.py)
├── pwa/                 # Propagator reference functions (BW, flatte980, flatte1270, flatte500)
├── data/                # Small real-data samples (100 events, incomplete — missing amp tensors)
├── tests/               # test_integration.py (53 tests), test_codegen_numeric.py (40 tests),
│                        # test_ir_snapshot.py (24 tests)
└── .llm_cache/          # by_hash/<sha256>/ + golden/<artifact>/latest -> by_hash symlink
```

## Key Files

- `create_all_scripts.py` — Main entry point: runs the full pipeline
- `create_code/create_control.py` — Orchestrator: iterates over generator_kk.json modules, calls codegen
- `create_code/ir/builder.py` — 806-line rule engine: config → PWAIR (deterministic, no LLM)
- `create_code/codegen/artifact_registry.py` — 17 artifact specs (CodeScripts + RunScripts + tensor)
- `create_code/cache/store.py` — Cache key = sha256(IR + prompt_ver + model + artifact + python_minor)
- `rendered_scripts/fit_object_kk.py` — Main generated code: PWAFunc + Control class (~450 lines)
- `config/pwa_info_kk.json` — Physics model: 6 resonances, 59 free parameters

## Data Conventions

- **dplex complex numbers**: `(2, ...)` shape — axis 0 = (real, imag)
- **Amplitude arrays**: `(damp, n_events, 2)` — damp FIRST, events, then real/imag pair
- **Sbc arrays** (helicity angles): `(n_events,)` — 1D per event
- **Argument order in calculate_* methods**: Sbc-first — `(..., sbc_phi, sbc_f, amp)` — never amp-first
- **const/theta reshaping**: `np.array([...]).reshape(-1, damp)` — row per resonance sub-component

## Propagator Patterns

All BW helper methods use: `vmap(partial(self.<f_func>, Sbc=f), out_axes=1)(*f_paras)` — never `np.moveaxis(vmap(...), 1, 0)`.

```python
def BW_<name>(self, phi_paras, f_paras, sbc_phi, sbc_f):
    a = self.BW(phi_mass, phi_width, sbc_phi)
    b = vmap(partial(self.<f_func>, Sbc=sbc_f), out_axes=1)(*f_paras)
    return dplex.deinsum("j,ij->ij", a, b)
```

## S6a JAX Optimization Rules (enforced in CI)

1. **Static tensors as module-level constants** — no `onp.eye/zeros` inside `@jit`
2. **vmap with `out_axes=1`** — never `moveaxis(vmap(...), 1, 0)`
3. **Always `jit(func, device=self.device)`** — never bare `jit(func)`

## Build & Test Commands

```bash
# Full pipeline (with LLM, needs DEEPSEEK_API_KEY)
python create_all_scripts.py

# Full pipeline (offline — uses golden cache, no API key needed)
unset DEEPSEEK_API_KEY && python create_all_scripts.py

# Run all tests (117 currently passing)
python -m pytest tests/ -v

# Run specific test files
python -m pytest tests/test_integration.py -v
python -m pytest tests/test_codegen_numeric.py -v

# Demo (standalone, in /tmp/pwacg_demo/)
python /tmp/pwacg_demo/run_demo.py       # Likelihood + gradient checks
python /tmp/pwacg_demo/run_fit_demo.py   # Full BFGS minimization
```

## Cache & Offline Operation

- Cache hit = 0 tokens, <1ms. Cache miss = LLM call → store → promote_to_golden.
- Golden fallback via symlink: `.llm_cache/golden/<artifact>/latest -> ../../by_hash/<hash>/`
- Project is fully offline-buildable with golden entries present.
- New resonance type → LLM generates → validate → store in `by_hash/` → promote symlink.
- Prompt changes → new prompt version → all hash keys change → re-generation triggered.

## 17 Artifacts (from generator_kk.json routing)

```
jinja_fit_info keys:  fit, batch, lasso, pull     → CodeScript (rendered_scripts/) + RunScript (run/)
jinja_draw_info keys: select, draw_lh, draw_wt, dplot → CodeScript + RunScript
Special:              tensor                       → run/RunCacheTensor.py
```

CodeScript artifact_name = module key (e.g. "fit"). RunScript artifact_name = key + "_run" (e.g. "fit_run").
