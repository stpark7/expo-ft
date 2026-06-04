# Simulation client (RoboCasa365)

The actor for **simulation** EXPO-FT. It runs the RoboCasa365 rollout environment plus the
env-agnostic WebSocket server (`client/run_client.py`), so the learner can drive
`reset` / `step` / `get_observation` over the network — exactly the same boundary as the real
client, just backed by a simulator instead of a robot.

> Install lives in the main [Setup](../../README.md#simulation-robocasa365--clientsim): this venv,
> the RoboCasa365 + robosuite forks, one-time `setup_macros` / kitchen-asset download, and the
> pi0.5 checkpoint + demo-data download. The shared RL pipeline — data conversion, SFT, EXPO-FT
> training, evaluation — lives in [Running Experiments](../../README.md#running-experiments-with-droid--pi05).
> This page covers only the **simulation-specific** operation.

## How sim differs from the real client

- **No robot hardware** — no NUC, no DROID server, no ZED/spacemouse. Nothing to configure under
  `droid/misc/parameters.py`.
- **Success & reward come from the simulator.** RoboCasa supplies them through its own task
  checker, so the sim client uses **no vision success detector** (`client/real/real_utils/detector.py`
  has no sim equivalent — don't add one).
- **Env wrapper** is `client/sim/envs/robocasa_env.py` (vs `client/real/envs/droid_env.py`).
- **Demo data** is downloaded with RoboCasa's own script (`scripts/sim/setup_demo_data.sh`), not
  collected by teleoperation.

## Adding a task (sim)

1. **Environment class** — add/extend a wrapper in `client/sim/envs/robocasa_env.py` for your
   RoboCasa task (observation space, action space, reset). Read success/reward straight from the
   simulator; do not add a detector.
2. **Task config** — add `configs/task/<task>.py` (bounds, language instruction, `control_hz`,
   `residual_action_xyzg`). The language instruction is used as the VLA prompt.
3. **Scripts** — only `scripts/pick/` (real) is fully wired in this repo, and `scripts/sim/`
   currently holds just `setup_demo_data.sh`. To run a full sim experiment, copy the relevant
   `scripts/pick/` steps, point the client step at `client/sim/.venv`, and update the hand-wired
   dataset / checkpoint / asset paths.

## Run the rollout client

The sim client uses the **same** entrypoint as the real one, just from the sim venv:

```bash
# On the machine running the simulator.
source client/sim/.venv/bin/activate
python -m client.run_client \
    --server_host=0.0.0.0 \
    --server_port=8102 \
    --config_task_path=configs/task/<your_sim_task>.py
```

- `--server_host` — `0.0.0.0` accepts remote connections from the learner
- `--server_port` — keep aligned with the SSH tunnel and the learner's `client_port`
- `--config_task_path` — sim task config

When the learner is on a different machine, set up the SSH reverse tunnel with
`scripts/set_server.sh` (see the main README), then start the learner with the shared
`run_server.sh` / `run_server_async.sh` from [Running Experiments](../../README.md#4-expo-ft-finetuning).

## Invariants worth knowing

- **`config_task` must be byte-identical on client and server** — both load it independently; it
  defines obs/action shapes and the language prompt.
- **`control_hz` is an implicit contract** between the learner's step-timing loop and this client.
- **Client recovery** — stop and restart only the client on an error; the learner waits for the
  connection to recover and continues.
