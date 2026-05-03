# FlowIS Refactor Playbook (Live)

This file is the single source of truth for execution rules and the live refactor plan.
Outdated items are replaced, not appended.

Last updated: 2026-04-22 (Asia/Shanghai) - mode-split NDE/TDE + sparse weighted injection controls implemented

## 1) Hard Rules

- Use only `FlowIS` conda environment for project commands.
- Never validate in `base`.
- Prefer non-interactive execution:
  - `conda run -n FlowIS <command>`
- Editable install from repo root:
  - `conda run -n FlowIS python -m pip install -e .`

## 2) Runtime Prerequisites

- Notebook stack:
  - `conda run -n FlowIS python -m pip install -e ".[torch,experiments]"`
- If `rl_agents` extras are missing:
  - `conda run -n FlowIS python -m pip install tensorboardX docopt numba seaborn moviepy`
- Notebook preflight:
  - `conda run -n FlowIS python scripts/check_highd_setup.py`

## 3) Refactor Goal

Build a readable, visual end-to-end experiment notebook (HighD-style) that demonstrates:

- distribution fitting/generation,
- simulation testing with probability-driven scene/behavior injection,
- final experiment figure generation.

At the same time, unify FlowIS simulation so proposal sampling and importance reweighting are mathematically consistent and reproducible.

## 4) Scope

In scope:
- HighD and InD injection/reweighting logic.
- Distribution handoff between env and vehicle controllers.
- Path/config handling for model artifacts.
- Reproducible simulation entrypoint (script-level).
- Visual verification of how distributions generate scenes and drive vehicle behavior.
- Notebook-first presentation of the complete pipeline (fit/generate -> simulate -> plot).

Out of scope (for this pass):
- Redesigning research metrics.
- Replacing all legacy notebook visualization code.
- Full data format migration away from existing `.joblib`.

## 5) Live Plan Board

Legend: `todo` | `doing` | `done` | `blocked`

1. Audit current injection and IS consistency  
Status: `done`

2. Define target contract for sampling/reweight interfaces (`p`, `q`, `logw` ownership)  
Status: `done`

3. Refactor env/vehicle code to use explicit proposal-target channels  
Status: `done`

4. Move per-trajectory weight accumulation from notebook ad-hoc code to reusable API  
Status: `todo`

5. Add deterministic smoke tests (env make/reset/step + weight sanity checks)  
Status: `done`

6. Add runnable script for HighD simulation reproduction (minimal CLI/config)  
Status: `done`

7. Update docs and notebook notes to point to new pipeline  
Status: `todo`

8. Build final end-to-end notebook template with visual checkpoints at each stage  
Status: `todo`

9. Add explicit visualization cells for scene sampling, conditional behavior sampling, and per-step injected controls  
Status: `doing`

10. Add HighD cut-in accelerated-testing mode (attacker vehicle targets configurable victim lane)  
Status: `doing`

11. Expose cut-in mode in notebook one-click interfaces  
Status: `doing`

## 5.1) Progress Notes (Current Session)

- Completed:
  - Fixed `FlowIS/vehicle/sampling.py` encoding and made failure path explicit (`return None, None, max_trials`).
  - Added proposal/target channel separation in FlowIS behavior controller with per-step and per-episode log-weight tracking.
  - Exposed FlowIS trace fields through env `info` for notebook-side visualization.
  - Added deterministic smoke script `scripts/check_flowis_injection.py` to validate accept/fallback paths and config keys.
  - Added runnable trace-export script `scripts/run_highd_flowis_trace.py` (CLI) for quick rollout diagnostics.
  - Added notebook template `Experiments/HighD_flowis_pipeline.ipynb` with trace collection and visualization cells.
  - Removed hardcoded HighD artifact paths from env; switched to config-driven artifact resolution.
  - Implemented sparse critical-event injection for HighD:
    - non-critical steps fallback to IDM with zero IS increment,
    - critical steps resample behavior at configurable decision intervals,
    - hold previous injected behavior between decision points with zero IS increment.
 - Added HighD config knobs for sparse injection:
    - `flowis_event_mode`, `flowis_ttc_threshold`, `flowis_distance_threshold`,
      `flowis_delta_v_threshold`, `flowis_decision_interval`.
  - Added mode-aware artifact loading and config overrides:
    - `flowis_tde_follow_path`, `flowis_nde_follow_path`,
      `flowis_tde_cutin_path`, `flowis_nde_cutin_path`.
  - Added sparse weighted-injection budget controls to reduce IS variance:
    - `flowis_max_weighted_steps`, `flowis_max_weighted_steps_follow`,
      `flowis_max_weighted_steps_cutin`, `flowis_importance_weighting`.
  - Upgraded `FlowIS_Follow` sampling to support both 1D and 2D behavior injection:
    - auto-selects rejection sampler (`1D`/`2D`) from distribution dims,
    - carries per-step lateral behavior (`behavior_steering`) for cut-in control,
    - forces unweighted Monte Carlo behavior when `proposal == target` (NDE rollout).
  - Exposed weighted-step counters in env trace info:
    - `weighted_steps`, `max_weighted_steps`.
  - Verified compile and key imports in `FlowIS` conda env.
  - Added HighD `flowis_attack_mode` with `follow`/`cutin` switch in env config.
  - Implemented cut-in attack scene construction:
    - attacker spawned on adjacent lane,
    - victim selectable (`flowis_cutin_target`: `background` or `ego`),
    - lane-merge commit trigger by longitudinal proximity.
  - Extended `FlowIS_Follow` with optional target-vehicle tracking and cut-in commit controls.
  - Updated notebook GIF interface with attack-mode parameters:
    - `attack_mode`, `cutin_target`, `cutin_side`.
- In progress:
  - Validate cut-in path under wider external artifact schemas (different scene feature definitions).
  - Add dedicated notebook cells to visualize cut-in commit timing and per-step importance weights.

## 6) Acceptance Criteria

- `HighDEnv-v0` and `InDEnv-v0` instantiate in `FlowIS` env when artifacts are configured.
- Sampling distribution used in behavior generation matches reweighting assumptions.
- A single command can run a short simulation batch and produce result artifacts.
- Preflight + smoke checks pass in `FlowIS` env.
- A single notebook can narrate and visualize the full experiment workflow from distribution construction to final figures.
- Distribution-to-scene and distribution-to-behavior injection can be visually validated (not only by logs).

## 7) Known Risks

- Historical `.joblib` objects may encode old schema assumptions.
- Hardcoded artifact paths still exist in legacy env files.
- Mixed installed `highway_env` packages can cause duplicate registration warnings.
