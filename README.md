# EXPO-FT: Sample-Efficient Reinforcement Learning Finetuning for Vision-Language-Action Models

Code for the paper *"EXPO-FT: Sample-Efficient Reinforcement Learning Finetuning for Vision-Language-Action Models"*

**[Project Website](https://pd-perry.github.io/expo-ft)** | **[arXiv](https://arxiv.org/abs/2605.25477)**

## Setup

The repo has **multiple independent Python environments that must never be merged** (their pins conflict):

- **Server (learner)** — `.venv/` at the repo root, managed by `pyproject.toml` + `uv.lock`. Holds the modern jax / openpi / lerobot stack used for RL training.
- **Client (actor)** — one self-contained venv per environment domain. Pick the one matching your setup; both run the same env-agnostic rollout server (`client/run_client.py`):
  - **Real robot (DROID)** — `client/real/.venv`, managed by `client/real/pyproject.toml` + `client/real/uv.lock`. Holds DROID's older numpy / mujoco / opencv pins for the real-robot SDK.
  - **Simulation (RoboCasa365)** — `client/sim/.venv`, managed by `client/sim/pyproject.toml` + `client/sim/uv.lock`. Holds the RoboCasa365 simulation stack (Python 3.11, modern mujoco). RoboCasa365 is built on **robosuite**, which it pulls in as its simulation backend.

Both require **Python 3.11+** and [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Clone the forks

EXPO-FT depends on several GitHub forks, all installed as **editable local checkouts** (see the `[tool.uv.sources]` blocks in each `pyproject.toml`). Clone the ones you need **before running `uv sync`** — uv fails if a referenced checkout is missing. You only need the forks for the environments you actually run (e.g. skip DROID if you only use simulation).

| Fork | Clone into | Used by |
| --- | --- | --- |
| [modified OpenPI](https://github.com/pd-perry/openpi/tree/expo_ft) | `expo_ft/agents/vla/openpi` | server **and** every client (provides `openpi-client`) |
| [DROID](https://github.com/pd-perry/droid) | `client/real/droid` | real-robot client only |
| RoboCasa365 | `client/sim/robocasa365` | simulation client only |
| [robosuite](https://github.com/ARISE-Initiative/robosuite) (master) | `client/sim/robosuite` | RoboCasa365's simulation backend (dependency) |

```bash
# From the repo root.

# Shared by the server and every client.
git clone -b expo_ft https://github.com/pd-perry/openpi.git expo_ft/agents/vla/openpi

# Real-robot client (DROID).
git clone https://github.com/pd-perry/droid.git client/real/droid

# Simulation client (RoboCasa365) and its robosuite backend
git clone https://github.com/ARISE-Initiative/robosuite.git client/sim/robosuite
git clone https://github.com/robocasa/robocasa.git client/sim/robocasa365
```

### Server (Learner)

Installs all server dependencies — including the local `expo_ft/agents/vla/openpi` checkout (editable) — via uv:

```bash
# From the repo root.
uv sync
```

### Client (Actor)

The actor runs the rollout environment plus the env-agnostic WebSocket server (`client/run_client.py`, shared by both clients). Install **only** the client that matches your setup — each lives in its own venv and must never be merged with the other or with the server. Both depend on the `openpi-client` package from the `expo_ft/agents/vla/openpi` checkout, so that fork must be cloned regardless (see [Clone the forks](#clone-the-forks)).

#### Real robot (DROID) — `client/real`

Installs the local `client/real/droid` checkout (editable) + `openpi-client` into `client/real/.venv`.

System prerequisites (install **before** `uv sync`):

- **ZED SDK** (only if you use ZED cameras). Install from [stereolabs.com](https://www.stereolabs.com/developers/release/) — provides the system libraries that `pyzed` loads at runtime.
- **Spacemouse HID access** — see the [PySpaceMouse troubleshooting guide](https://github.com/JakubAndrysek/PySpaceMouse/blob/master/troubleshooting.md) for platform-specific setup. On Linux, you need a udev rule.

Install:

```bash
# From the repo root.
# 1. Install real-robot client dependencies into ./client/real/.venv.
cd client/real && uv sync && cd ../..

# 2. (Optional) Install pyzed if you use a ZED camera. Must be a separate step
#    because the pyzed wheel's numpy>=2.0 metadata over-constrains a binary
#    that actually works against numpy 1.x, so we bypass uv's resolver.
bash client/real/install_pyzed.sh
```

#### Simulation (RoboCasa365) — `client/sim`

Installs the local `client/sim/robocasa365` checkout (editable) — together with its robosuite backend (`client/sim/robosuite`) — plus `openpi-client` into `client/sim/.venv`. No ZED/spacemouse hardware is needed: RoboCasa supplies success and reward through its own task checker, so the sim client uses no vision detector.

```bash
# From the repo root.
# 1. Install simulation client dependencies into ./client/sim/.venv.
cd client/sim && uv sync && cd ../..

# 2. One-time RoboCasa setup + kitchen asset download (~10GB).
source client/sim/.venv/bin/activate
python -m robocasa.scripts.setup_macros
python -m robocasa.scripts.download_kitchen_assets
deactivate
```

#### RoboCasa365 pi0.5 checkpoint & demo data

EXPO-FT finetunes *from* the pretrained robocasa365 pi0.5 checkpoint (multitask, human-300) rather than training a policy from scratch — download it before running. The demo data is optional (only needed to seed the offline replay buffer; the normalization stats RL needs already ship inside the checkpoint's `assets/`).

**Checkpoint** (~45 GB) — grabs only the `pi05_pretrain_human300/multitask_learning/75000` subfolder (`params/` + `train_state/` + bundled norm-stats `assets/`) from [`robocasa/robocasa365_checkpoints`](https://huggingface.co/robocasa/robocasa365_checkpoints) into `./checkpoints/robocasa365/` (git-ignored):

```bash
# Any venv with the `hf` CLI works; the sim venv also has hf_transfer for faster downloads.
source client/sim/.venv/bin/activate
HF_HUB_ENABLE_HF_TRANSFER=1 hf download \
    robocasa/robocasa365_checkpoints \
    --include "pi05_pretrain_human300/multitask_learning/75000/*" \
    --local-dir ./checkpoints/robocasa365
deactivate
# Lands at: ./checkpoints/robocasa365/pi05_pretrain_human300/multitask_learning/75000/{params,train_state,assets}
```

**Demo data** (optional) — a helper script downloads it into the repo's `data/` dir (git-ignored), matching the repo's dataset convention. By default it fetches just the target task (both splits) instead of all 300 pretraining tasks:

```bash
bash scripts/sim/setup_demo_data.sh
# Lands at: data/robocasa365/v1.0/{pretrain,target}/.../PickPlaceCounterToStove/...
```

RoboCasa's downloader has no output-dir flag — it reads the save location from `DATASET_BASE_PATH` in `client/sim/robocasa365/robocasa/macros_private.py`. The script sets that to `<repo>/data/robocasa365` for you (idempotent) and then downloads. To grab more / different data (other tasks, splits, or sources such as the full pretraining set or mimicgen), edit the `TASKS` / `SPLITS` / `SOURCE` vars at the top of the script — see the [dataset catalog](https://robocasa.ai/docs/build/html/datasets/using_datasets.html).

## Overview and Code Structure

The system uses a **server-client architecture**: the **server (learner)** runs RL training with the VLA policy, while the **client (actor)** runs the DROID real-robot rollout environment and communicates over WebSocket.

```
train_pi_robo.py                # Server: synchronous RL finetuning loop
train_pi_robo_async.py          # Server: asynchronous RL finetuning loop (sampler + updater on separate GPUs)
eval_droid_policy.py            # Server: standalone policy evaluation

client/
  run_client.py                 # Client: env-agnostic rollout server (WebSocket), shared by both clients
  common/                       # Shared helpers (image processing) used by every client env
  real/                         # Real-robot (DROID) client — own venv + droid fork
    README.md                   #   → Real-robot operation guide (hardware, data collection, run)
    collect_data.py             #   Demonstration data collection (spacemouse)
    envs/                       #   Environment wrappers (DROID real robot)
    real_utils/                 #   Success detectors, spacemouse, visualization
  sim/                          # Simulation (RoboCasa365) client — own venv + robocasa365 fork (+ robosuite backend)
    README.md                   #   → Simulation operation guide (RoboCasa env, run)
    envs/                       #   Environment wrappers (RoboCasa)

configs/
  task/                         # Task configs
  model/                        # Algorithm configs: expo_ft_pi_config.py (EXPOLearner), dagger_pi_config.py (BCLearner)

expo_ft/
  agents/
    README.md                   # → How to add your own algorithm / VLA backend
    alg/                        # RL algorithms (EXPO-FT, BC, base agent)
    vla/                        # VLA wrappers (pi0.5 integration)
  data/                         # Replay buffer, dataset loading, batch processor
  env/                          # Server-side env utilities (WebSocket client, dataset loading)
  networks/                     # Neural network components (encoders, critics, MLP)
  distributions/                # Action distributions (tanh normal)
  utils/                        # Logging, training utilities, augmentation

scripts/                        # Shell scripts for launching experiments
  convert_droid_data_to_lerobot.py
  set_server.sh
  pick/                         # Complete example script set for the pick task
```

## Use your own algorithm & VLA

EXPO-FT is built around two pluggable layers: the **algorithm** (learner) in `expo_ft/agents/alg/` and the **VLA backend** (policy) in `expo_ft/agents/vla/`. You can swap either without touching the training loop — implement the learner / VLA API, expose it through a `configs/model/` config, and (for a new algorithm) add a `model_cls` branch in the training scripts.

See **[`expo_ft/agents/README.md`](expo_ft/agents/README.md)** for the learner API, the EXPO-FT / BC internals, and step-by-step instructions for adding a new algorithm or VLA.

## Running Experiments with DROID + pi0.5

The steps below are the **shared, server-side pipeline**: data conversion → SFT → EXPO-FT training → evaluation. The **domain-specific actor operation** — hardware setup, demo collection, success detection, and launching the rollout client — lives in the client READMEs:

- **Real robot (DROID)** → [`client/real/README.md`](client/real/README.md)
- **Simulation (RoboCasa365)** → [`client/sim/README.md`](client/sim/README.md)

### OpenPI Setup

We use a [modified fork of OpenPI](https://github.com/pd-perry/openpi/tree/expo_ft) with support for frozen encoder training (for efficient action sampling) and Cartesian action control for DROID. Cloned into `./expo_ft/agents/vla/openpi` and installed editable during the [server setup](#server-learner) step (see [Clone the forks](#clone-the-forks)). The same checkout provides the SFT pretraining scripts wrapped below.

### Task setup

Three task-specific pieces feed the pipeline; create them once per task (real/sim details are in the client READMEs):

1. **Environment class** — `client/real/envs/droid_env.py` (real) or `client/sim/envs/robocasa_env.py` (sim), matching your task's observation/action space and reset behavior.
2. **Task config** — `configs/task/<task>.py` (bounds, reset joints, language instruction, camera serials, `control_hz`, `residual_action_xyzg`). See `configs/task/pick.py`.
3. **Success detector** — real only, in `client/real/real_utils/detector.py`, registered in the env's `detect()`; sim reads success/reward straight from the simulator.

> The same `config_task` must be **byte-identical on client and server** — both load it independently, and it defines observation/action shapes and the VLA prompt.

### Running the Experiment

All commands below use `scripts/${TASK_NAME}/...`; refer to `scripts/pick/` for a complete working example of the task scripts. Only `scripts/pick/` is fully wired in this repo — for a new task, copy that directory and update the dataset paths, task config, checkpoint paths, and OpenPI asset IDs.

> **Filesystem note:** The example scripts assume the client and server can see the same repo-relative paths. If they run on different filesystems, collect data on the client/robot machine, then copy or sync the collected `data/...` directory to the server/GPU machine before running conversion, norm stats, SFT, RL training, or evaluation. The `dataset_path`, OpenPI assets, SFT checkpoints, and EXPO-FT checkpoints in the server scripts are server-local paths. The client only needs the robot environment code, task config, and network access to the learner; keep any task config or environment changes synced on both machines.

#### 1. Collect demonstration data

Domain-specific — see the client READMEs:

- **Real robot** — teleoperate with a spacemouse (NUC DROID server must be up): [`client/real/README.md`](client/real/README.md#data-collection-spacemouse).
- **Simulation** — download RoboCasa demos with the helper script: [demo data in Setup](#robocasa365-pi05-checkpoint--demo-data).

#### 2. Data Conversion

Convert collected data to LeRobot format for pi0.5 finetuning:

```bash
# On the server / GPU machine.
bash scripts/${TASK_NAME}/convert_data.sh
```

Parameters to update in `convert_data.sh`:

- `MAX_EPISODES` -- max number of collected episodes to convert
- `TASK_CONFIG` -- task config used to interpret the raw DROID data
- `DATA_DIR` -- source directory containing successful demonstrations
- `REPO_NAME` -- LeRobot dataset repo/id written into the converted dataset

#### 3. Policy Pretraining (SFT)

Before RL finetuning, pretrain the policy with supervised finetuning on the collected demonstrations.

Both scripts are thin wrappers around the OpenPI training entrypoints (`expo_ft/agents/vla/openpi/scripts/compute_norm_stats.py` and `expo_ft/agents/vla/openpi/scripts/train.py`), so they require the OpenPI checkout from [Clone the forks](#clone-the-forks). Run them from the repo root with the server `.venv` active.

**3.1 Calculate normalization statistics** (first time only for a new task):

```bash
# On the server / GPU machine.
bash scripts/${TASK_NAME}/calculate_norm.sh
```

Parameters to update in `calculate_norm.sh`:

- `REPO_ID` -- LeRobot dataset id from the conversion step

> **Fixed-state tasks:** update the state and action standard deviations in
> the OpenPI config after computing normalization stats. Use the q01/q99 values
> and set the standard deviation to `1`.

**3.2 Finetune pi0.5**:

```bash
# On the server / GPU machine.
bash scripts/${TASK_NAME}/finetune_droid.sh
```

Parameters to update in `finetune_droid.sh`:

- `DATA_ID` / `REPO_ID` -- dataset id used for the converted LeRobot data
- `ASSETS_DIR` / `ASSET_ID` -- OpenPI normalization assets from the stats step

The provided pick script runs about 4000 steps, which was sufficient for all tasks we tested.

#### 4. EXPO-FT Finetuning

After pretraining, finetune the policy with EXPO-FT. The server (learner) runs the RL training loop and the client (rollout server) runs the environment; they communicate over WebSocket.

**Start the rollout client** the way that matches your setup:

- **Real robot** — start the DROID server on the NUC, then run the client: [`client/real/README.md`](client/real/README.md#run-the-rollout-client).
- **Simulation** — launch the sim rollout client: [`client/sim/README.md`](client/sim/README.md#run-the-rollout-client).

**If the server and client are on different machines**, set up an SSH reverse tunnel on the client machine so the server can reach it via `localhost`:

```bash
# On the client machine, forward port to the server machine
bash scripts/set_server.sh <server-hostname> 8102 <your-username>
```

Arguments to update in `set_server.sh`:

- `<server-hostname>` -- GPU training machine reachable over SSH
- `8102` -- port to forward; keep aligned with `run_policy.sh` and learner `client_port`
- `<your-username>` -- SSH username on the training machine

**Then start the server** (on the GPU training machine):

```bash
# On the server / GPU machine.
bash scripts/${TASK_NAME}/run_server.sh        # synchronous
bash scripts/${TASK_NAME}/run_server_async.sh   # asynchronous
```

> **Async training:** Requires ≥ 2 GPUs (1 sampler + ≥ 1 updater). Use it when one episode takes a long time; otherwise synchronous training can yield better results.

Key parameters to configure in `scripts/${TASK_NAME}/run_server.sh` and `scripts/${TASK_NAME}/run_server_async.sh`:

- `dataset_path` -- path to the collected demonstration data
- `num_data` -- max offline demo episodes to seed into the replay buffer (0 = all)
- `update_type` / `num_updates` -- for synchronous training, recommend: use episode updates with `env_steps / num_updates` close to 20-30
- `edit_scale` -- residual edit scale; 0.2 is a good starting point
- `client_host` / `client_port` -- set to `localhost` / `8102` when using SSH tunnel or running on the same machine

> **Client recovery:** If the client hits an error or the robot gets stuck during online training, you can stop and restart only the client. The server waits for the policy/environment connection to recover, then continues training once the client is restarted.

#### 5. Evaluation

Evaluate the trained policy. Start the DROID server and the client rollout server the same way as in [EXPO-FT Finetuning](#4-expo-ft-finetuning), then launch evaluation from the server/GPU machine. All model parameters should match the training configuration.

```bash
# On the server / GPU machine.
bash scripts/${TASK_NAME}/eval_policy.sh
```

Parameters should match the corresponding `run_server.sh` or `run_server_async.sh` training settings.

## Citation

```bibtex
@misc{dong2026expoft,
      title={EXPO-FT: Sample-Efficient Reinforcement Learning Finetuning for Vision-Language-Action Models},
      author={Perry Dong and Kuo-Han Hung and Tian Gao and Dorsa Sadigh and Chelsea Finn},
      year={2026},
      eprint={2605.25477},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2605.25477},
}
```

