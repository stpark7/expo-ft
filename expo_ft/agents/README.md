# Agents: algorithms & VLA backends

This directory holds the two pluggable layers of EXPO-FT:

- `alg/` — the **online fine-tuning algorithms** (the learner). `expo_ft.py` (`EXPOLearner`) is the main method; `bc.py` (`BCLearner`) is the DAgger baseline; `agent.py` defines the shared API both implement.
- `vla/` — the **VLA backends** (the policy). `vla_base.py` is the backend-agnostic contract; `pi05.py` is the only current implementation (OpenPI pi0.5).

The training scripts (`train_pi_robo.py`, `train_pi_robo_async.py`) are written against these two abstractions, so you can swap either layer without touching the training loop.

> This README is the place to grow as new algorithms / VLAs land. When you add one, document its config, its invariants, and anything non-obvious here.

---

## The learner API (`alg/agent.py`)

Every algorithm is a Flax `struct.PyTreeNode` exposing the same `AgentLearner` surface, so the training scripts can treat them interchangeably:

| Method | Role |
| --- | --- |
| `create(...)` | Build the learner (networks, optimizers, target params) from a model config. |
| `sample_actions(obs, ...)` | Produce the action to execute at a control step (the actor / rollout path). |
| `update(batch, ...)` | One gradient step from replay data (critic / actor / temperature as applicable). |
| `update_actor(batch, ...)` | The VLA action-expert update (separated out because it has its own batch + cadence). |

### `EXPOLearner` (`expo_ft.py`)

Per control step, `sample_actions` runs a **propose → edit → select** pipeline:

1. The frozen-encoder **pi0.5 base policy** samples `N` candidate action *chunks* (config `N`, default 8).
2. A **residual actor** (`PixelEditMultiplexer(TanhNormal(...))`) proposes edits to the first `n_edit_samples` candidates, scaled by `edit_scale` (default 0.2) and — when `residual_action_xyzg=True` — masked to xyz + gripper only (rotation dims 3–5 zeroed).
3. A **Q-ensemble critic** (`num_qs=10`, RedQ-style `num_min_qs=2` subsample) scores all `N + n_edit_samples` candidates over a `BatchEncoder` (ResNetV2 per camera) embedding; `argmax Q` selects the action actually executed.

`update` trains three things from replay data: the **critic** (n-step Bellman target over `replan_steps`, soft target update `tau`), the **residual actor + temperature** (SAC-style entropy-regularized Q maximization), and the **pi0.5 base actor** via `update_actor`. The pi0.5 *vision encoder* is frozen (`freeze_pi05_encoder=True`) for sampling efficiency, but its action expert is still finetuned — by default only on **successful** transitions (`actor_success_only=True`). `utd_ratio` (default 20) splits each batch into minibatches scanned through the critic update.

### `BCLearner` (`bc.py`)

The DAgger baseline: same pi0.5 actor, **no critic / no residual**, trained only on human-intervention chunks. Interchangeable with `EXPOLearner` through the shared API.

---

## Add your own algorithm

1. **Implement the learner** in `alg/your_alg.py` as a `struct.PyTreeNode` exposing the `AgentLearner` API (`create`, `sample_actions`, `update`, `update_actor`). Use `EXPOLearner` / `BCLearner` as references.
2. **Add a model config** in `configs/model/` (see [Config system](#config-system) below) and set its `model_cls` string to your class name.
3. **Register the dispatch.** The training scripts pick the learner with a hardcoded `if/elif` on `config.model_cls` — there is no registry. Add a branch in `train_pi_robo.py` (and `train_pi_robo_async.py` if you want async support):

   ```python
   if config.model_cls == "EXPOLearner":
       ...
   elif config.model_cls == "YourLearner":   # <-- add this
       agent = YourLearner.create(...)
   ```

---

## Add your own VLA backend

1. **Implement the contract** in `vla/your_vla.py` following `vla_base.Model`: process raw obs → sample → unnormalize → format the batch for the actor loss. `pi05.py` is the reference (`build_pi05()` loads the OpenPI fork's pi0.5 model, train state, EMA target params, and wires JAX/FSDP sharding).
2. **Mind the transform pipeline.** Observations flow through OpenPI transform stages (repack → robot data transforms → **Normalize via norm stats** → resize/tokenize); actions are zero-**padded** to the model's action dim and unpadded back to the env's. Keep the replay buffer's transforms in sync (`expo_ft/data/` runs the *same* pipeline on insert).
3. **Expose it** so the learner can construct it (the learner builds its VLA in `create`).

---

## Config system (`configs/`)

`ml_collections.ConfigDict` files loaded via absl `config_flags`. Two independent flags select a run:

- `--config` → **model/algorithm** config. Linear inheritance: `td_config → sac_config → expo_ft_pi_config → dagger_pi_config`. The `config.model_cls` field is what the `if/elif` in the training scripts dispatches on.
- `--config_task` → **task** config (lives in `configs/task/`, see the client READMEs). Holds the env class, language prompt, bounds, camera serials, `control_hz`, `residual_action_xyzg`.

CLI overrides like `--config.N=8` work for scalars (`lock_config=False`) but **not** for numpy-array fields (`bounds`, `reset_joints`) — edit those in the config file.
