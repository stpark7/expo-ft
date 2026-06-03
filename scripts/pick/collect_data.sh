#!/usr/bin/env bash

source client/real/.venv/bin/activate

NUM_EPISODES=15

python -m client.real.collect_data \
    --save_root data/pick_cube_balance \
    --num_episodes $NUM_EPISODES \
    --task_config configs/task/pick.py
