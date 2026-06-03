#!/usr/bin/env bash

source client/real/.venv/bin/activate

python -m client.run_client \
    --server_host=0.0.0.0 \
    --server_port=8102 \
    --config_task_path=configs/task/pick.py
