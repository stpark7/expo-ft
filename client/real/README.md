# Real-robot client (DROID)

The actor for **real-robot** EXPO-FT. It runs the DROID rollout environment plus the
env-agnostic WebSocket server (`client/run_client.py`), so the learner on the GPU machine
can drive `reset` / `step` / `get_observation` over the network.

> Install lives in the main [Setup](../../README.md#real-robot-droid--clientreal) (this venv,
> the DROID fork, ZED/spacemouse prerequisites). The shared RL pipeline — data conversion,
> SFT, EXPO-FT training, evaluation — lives in [Running Experiments](../../README.md#running-experiments-with-droid--pi05).
> This page covers only the **real-robot–specific** operation.

## Hardware configuration

Set the hardware-specific values **before** running the client. See the
[DROID documentation](https://droid-dataset.github.io/droid/) for software, hardware, and calibration.

**Client / laptop**

- `client/real/droid/droid/misc/parameters.py`
  - `nuc_ip`
- `configs/task/real_base.py` (and per-task overrides, e.g. `configs/task/light2.py`)
  - `side_camera_id`
  - `wrist_camera_id`

**NUC**

- `droid/misc/parameters.py`
  - `sudo_password`
  - `robot_type`
  - `robot_serial_number`
- DROID / Polymetis hardware config
  - `robot_ip`

> **Camera serials live in two places** — `config.side_camera_id` / `wrist_camera_id` here and
> `droid/misc/parameters.py` on the NUC must agree.

## Start the DROID server (on the NUC)

The NUC must be running the DROID server before data collection or rollouts:

```bash
# On the NUC, from the DROID install root (e.g. client/real/droid here, or the NUC's
# standalone DROID checkout).
python scripts/server/run_server.py
```

## Data collection (spacemouse)

Collect demonstrations with a spacemouse (NUC DROID server must be up):

```bash
# On the client / robot machine.
bash scripts/${TASK_NAME}/collect_data.sh
```

Parameters to update in `collect_data.sh`:

- `--save_root` — output directory for collected episodes
- `--num_episodes` — number of demonstrations to collect
- `--task_config` — task config for the robot environment

Only `success/` episodes are used downstream (conversion filters out pre-movement idle frames
via the `movement_enabled` flag).

## Adding a task (real)

1. **Environment class** — create one in `client/real/envs/droid_env.py` matching your task's
   observation space, action space, and reset behavior. The pick env is the reference.
2. **Task config** — add `configs/task/<task>.py` (bounds, reset joints, language instruction,
   camera serials, `control_hz`, `residual_action_xyzg`). See `configs/task/pick.py`.
3. **Success detector** — define it in `client/real/real_utils/detector.py` (e.g.
   `PickBlocksDetector`) and register it in the env class's `detect()` method.
4. Copy `scripts/pick/` to `scripts/<task>/` and update the hand-wired paths.

## Run the rollout client

```bash
# On the client / robot machine.
bash scripts/${TASK_NAME}/run_policy.sh
```

`run_policy.sh` activates `client/real/.venv` and launches the shared
`python -m client.run_client`. Parameters to update:

- `--server_host` — interface the rollout server binds to; `0.0.0.0` accepts remote connections
- `--server_port` — rollout server port; keep aligned with the tunnel and learner `client_port`
- `--config_task_path` — task config for the rollout environment

When the learner is on a different machine, set up the SSH reverse tunnel with
`scripts/set_server.sh` (see the main README). Then start the learner with `run_server.sh` /
`run_server_async.sh` as described in [Running Experiments](../../README.md#4-expo-ft-finetuning).

## Invariants worth knowing

- **`config_task` must be byte-identical on client and server** — both load it independently; it
  defines obs/action shapes, bounds, and the language prompt. If the machines don't share a
  filesystem, sync the task config (and collected `data/`) by hand.
- **`control_hz` (default 10) is an implicit contract** between the server's step-timing loop and
  this client; nothing asserts they match.
- **Client recovery** — if the client errors or the robot gets stuck mid-training, stop and
  restart **only** the client. The learner waits for the connection to recover and continues
  (`EnvClientWrapper._call` recreates the env, resets, and retries).
