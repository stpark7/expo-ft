#!/usr/bin/env bash
#
# RoboCasa365 시뮬레이션 롤아웃 클라이언트(actor)를 띄운다.
#
#   server_host=0.0.0.0  : learner의 원격 접속을 허용
#   server_port=8102     : SSH 리버스 터널 / learner의 --client_port와 맞출 것
#   config_task_path     : sim task config (client와 server에서 byte-identical)
#   fixed_layout_id/style_id : 시뮬레이터의 주방 레이아웃과 스타일을 고정한다. (강화학습 시 학습변수를 줄이기 위해)
#

source client/sim/.venv/bin/activate

python -m client.run_client \
    --server_host=0.0.0.0 \
    --server_port=8102 \
    --config_task_path=configs/task/robocasa.py \
    --fixed_layout_id=11 \
    --fixed_style_id=14
