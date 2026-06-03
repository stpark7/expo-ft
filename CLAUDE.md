# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code for the paper *EXPO-FT: Sample-Efficient Reinforcement Learning Finetuning for Vision-Language-Action Models*. Despite the `expo-ft` directory name, this is **not** an Expo / React-Native project — "EXPO-FT" is the RL algorithm. It finetunes a pi0.5 VLA policy on a real DROID robot using online RL, with a **server (learner) / client (actor)** split that communicate over WebSocket.

There is no test suite, linter, or CI in this repo. It is run via the shell scripts in `scripts/`, not a package test harness.

## Environments + forks (critical setup detail)

The repo has **multiple independent `uv` virtualenvs that must never be merged**, because the modern JAX/OpenPI learner stack, DROID's older numpy/mujoco/opencv stack, and the RoboCasa365 sim stack have mutually conflicting pins:

- **Server (learner)** — `.venv/` at repo root, from root `pyproject.toml` + `uv.lock`. Activate with `source .venv/bin/activate`.
- **Client (actor)** — one self-contained venv per environment domain; both run the shared `python -m client.run_client`:
  - **Real robot (DROID)** — `client/real/.venv`, from `client/real/pyproject.toml` + `client/real/uv.lock`. Older numpy/mujoco/opencv pins.
  - **Simulation (RoboCasa365)** — `client/sim/.venv`, from `client/sim/pyproject.toml` + `client/sim/uv.lock`. RoboCasa365 + its robosuite backend (numpy 2.x). RoboCasa supplies success/reward, so the sim client needs no vision detector.

They depend on **GitHub forks that are git-ignored and must be cloned before `uv sync`** (they install as editable local checkouts, so `uv sync` fails if absent):

```bash
git clone -b expo_ft https://github.com/pd-perry/openpi.git expo_ft/agents/vla/openpi   # used by ALL envs
git clone https://github.com/pd-perry/droid.git client/real/droid                       # real client only
git clone https://github.com/ARISE-Initiative/robosuite.git client/sim/robosuite        # sim: robocasa backend
git clone https://github.com/robocasa/robocasa.git client/sim/robocasa365               # sim client only
uv sync                                 # server env (repo root)
cd client/real && uv sync && cd ../..   # real client env
cd client/sim  && uv sync && cd ../..   # sim client env
```

`websockets~=13.1` is pinned **identically on both sides** — 14+ tightened handshake strictness and will break the server↔client connection if they drift.

## Running things

The canonical workflow is the ordered scripts in `scripts/pick/` (the only fully-wired task). Each is a thin wrapper that activates the right venv and calls a Python entrypoint with flags; you configure a run by **editing paths/flags inside the `.sh` file**, not via a config registry. See `README.md` for the full per-step parameter list. Pipeline order (artifacts of each step feed the next, paths are wired by hand):

```
collect_data.sh   →  convert_data.sh  →  calculate_norm.sh  →  finetune_droid.sh  →  run_server[.sh|_async.sh] + run_client  →  eval_policy.sh
 (client venv)        (server venv)       (server venv)          (SFT, server venv)   (server + robot machine)                  (server venv)
```

Server entrypoints (run from repo root, server venv active):
- `train_pi_robo.py` — **synchronous** RL loop (single mesh, rollout and updates interleaved on the main thread).
- `train_pi_robo_async.py` — **asynchronous** RL loop. Requires **≥2 GPUs**: `jax.devices()[0]` samples (actor, main thread), `[1:]` run gradient updates (learner, background daemon thread). New actor params are published to the sampler via an atomic reference swap of the inference cache. Use when one episode is slow; sync often trains better otherwise.
- `eval_droid_policy.py` — rollout-only evaluation from a checkpoint (no updates, no replay buffer).

Client entrypoint (robot machine, client venv active): `python -m client.run_client ...` — see `scripts/pick/run_policy.sh`.

When server and client are on different machines, the server connects to `localhost` and an SSH **reverse** tunnel (`scripts/set_server.sh`) forwards the client's port to it. Keep the port aligned across `run_client`, `set_server.sh`, and the learner's `--client_port` (default 8102).

## Architecture

### Server/client boundary
The training loop never touches robot hardware. `expo_ft/env/env_client.py` (`EnvClientWrapper` → `EnvClient`) presents a gym-like `reset/step/get_observation/get_info_for_step` interface but executes each as a **WebSocket RPC** to `client/run_client.py`, which owns the real `DroidEnv`. Note the inverted topology: the **client listens** on `0.0.0.0:8102`, the **server connects out** as a WebSocket client. Payloads are msgpack (`openpi_client.msgpack_numpy`). `EnvClientWrapper._call` transparently **recovers from a client crash/restart mid-training** — it recreates the env, resets, waits, and retries — which is why you can restart only the client without losing the server's training state.

### The EXPO-FT algorithm (`expo_ft/agents/alg/expo_ft.py`, `EXPOLearner`)
This is the heart of the system. Per control step, `sample_actions` runs a **propose → edit → select** pipeline:
1. The frozen-encoder **pi0.5 base policy** samples `N` candidate action *chunks* (config `N`, default 8).
2. A **residual actor** (`PixelEditMultiplexer(TanhNormal(...))`) proposes edits to the first `n_edit_samples` candidates, scaled by `edit_scale` (default 0.2) and — when `residual_action_xyzg=True` — masked to xyz + gripper only (rotation dims 3–5 zeroed).
3. A **Q-ensemble critic** (`num_qs=10`, RedQ-style `num_min_qs=2` subsample) scores all `N + n_edit_samples` candidates over a `BatchEncoder` (ResNetV2 per camera) embedding; `argmax Q` selects the action that is actually executed.

`EXPOLearner.update` trains three things from replay data: the **critic** (n-step Bellman target over `replan_steps`, soft target update `tau`), the **residual actor + temperature** (SAC-style entropy-regularized Q maximization), and the **pi0.5 base actor itself** via `update_actor`. The pi0.5 *vision encoder* is frozen (`freeze_pi05_encoder=True`) for sampling efficiency, but its action expert is still finetuned — by default only on **successful** transitions (`actor_success_only=True`). `utd_ratio` (default 20) splits each batch into minibatches scanned through the critic update.

The learner is a Flax `struct.PyTreeNode`; `_split_params`/`_merge_params` peel network params (and EMA params) out for Orbax checkpointing. `cache_infer_params()` pins inference params to one device to avoid per-rollout-step `device_put`.

`BCLearner` (`bc.py`) is the DAgger baseline: same pi0.5 actor, **no critic / no residual**, trained only on human-intervention chunks. The two learners are interchangeable through the shared `AgentLearner` API in `agent.py` (`create`, `sample_actions`, `update`, `update_actor`).

### VLA wrapper (`expo_ft/agents/vla/`)
`vla_base.py:Model` is the backend-agnostic contract (process raw obs → sample → unnormalize → format batch for actor loss). `pi05.py` is the only implementation; `build_pi05()` loads the OpenPI fork's pi0.5 model, train state, and EMA target params, and wires JAX/FSDP sharding. Observations flow through OpenPI transform stages (repack → robot data transforms → **Normalize via norm stats** → resize/tokenize), and actions are zero-**padded** to the model's action dim and unpadded back to the env's. To plug in a different VLA, implement `vla_base.Model` and expose it; to plug in a different algorithm, implement `AgentLearner` and add a `model_cls` branch (see config dispatch below).

### Config system (`configs/`)
`ml_collections.ConfigDict` files loaded via absl `config_flags`. Two independent flags select a run:
- `--config` → **model/algorithm** config. Linear inheritance: `td_config → sac_config → expo_ft_pi_config → dagger_pi_config`. The string field `config.model_cls` (`"EXPOLearner"` / `"BCLearner"`) is dispatched by a hardcoded `if/elif` in `train_pi_robo.py` (no registry — new algorithms require editing that branch).
- `--config_task` → **task** config. Inheritance: `real_base → {pick, light2}`. Holds the env class reference (`config.env`), `language_instruction` (used as the VLA prompt), workspace `bounds`, `reset_joints`, camera serials, `control_hz`, and `residual_action_xyzg`.

CLI overrides like `--config.N=8` work for scalars (`lock_config=False`) but **not** for numpy-array fields (`bounds`, `reset_joints`) — those must be edited in the config file.

### Data layer (`expo_ft/data/`)
`process_droid_dataset` (`env/droid_utils.py`) loads offline HDF5 demos into transition dicts. `PiReplayBuffer` stores transitions in fixed circular numpy arrays, running the **same OpenPI transform pipeline as the VLA** on insert (so buffered images are already normalized float32) and broadcasting each raw action into an `[action_horizon, action_dim]` chunk via a backward fill. `BatchProcessor` orchestrates two buffers — `replay_buffer` (online) and `offline_replay_buffer` — and assembles each update's **critic batch** and a separate **actor batch** (success-only for EXPO, HIL-only for BC). `--offline_ratio` controls the split: `0` (default) seeds demos straight into the *online* buffer; `>0` keeps a separate offline buffer mixed at that ratio.

## Cross-cutting invariants worth knowing

- **`config_task` must be byte-identical on client and server.** Both load it independently; it defines observation/action shapes, bounds, and the language prompt. A mismatch silently misaligns tensors or feeds the VLA the wrong prompt. If the two machines don't share a filesystem, sync the task config and the collected `data/` directory by hand.
- **Camera serials live in two places.** `config.side_camera_id`/`wrist_camera_id` (here) and `droid/misc/parameters.py` on the NUC must agree.
- **`control_hz` (default 10) is an implicit contract** between the server's step-timing loop and the client; nothing asserts they match.
- **Success detection is per-env-class, not a registry.** To add a task: subclass `DroidEnv`, override `detect()` / the detector init (`client/real/real_utils/detector.py`, e.g. `PickBlocksDetector`), point a new `configs/task/*.py` at it, and copy `scripts/pick/` updating all the hand-wired paths.
- **Only `success/` episodes are used** for conversion/SFT/RL; the conversion step filters out pre-movement idle frames via the `movement_enabled` flag.
- **Checkpoint paths are absolute and hand-wired** in each script (`pi05_weight_loader_path`, `pi05_assets_dir`/`asset_id`). There is no "use latest checkpoint" derivation — forgetting to update one loads a stale policy. `--resume`/`--overwrite` control existing checkpoint dirs; `--checkpoint_buffer` pickles replay transitions for exact resume.
- **The OpenPI fork is the source of truth for the VLA config** named by `pi05_config_name` (e.g. `expo_pi05_droid_lora_finetune_sft_cartesian_state`); SFT (`finetune_droid.sh`) and norm-stat (`calculate_norm.sh`) steps call OpenPI's own `train.py`/`compute_norm_stats.py` directly.
