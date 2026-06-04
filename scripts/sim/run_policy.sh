#!/usr/bin/env bash
#
# RoboCasa365 시뮬레이션 롤아웃 클라이언트(actor)를 띄운다.
#
# 실제 로봇용 scripts/pick/run_policy.sh와 동일한 entrypoint(client.run_client)를
# 쓰되, 활성화하는 venv가 client/sim/.venv 이고 task config가 sim 전용이라는
# 점만 다르다. 시뮬레이터가 도는 머신에서, **repo 루트에서** 실행한다.
#
#   server_host=0.0.0.0  : learner의 원격 접속을 허용
#   server_port=8102     : SSH 리버스 터널 / learner의 --client_port와 맞출 것
#   config_task_path     : sim task config (client와 server에서 byte-identical)
#
# client가 listen하고 server(learner)가 접속해 나오는 토폴로지다. learner가 다른
# 머신이면 scripts/set_server.sh로 리버스 터널을 먼저 연다.

source client/sim/.venv/bin/activate

python -m client.run_client \
    --server_host=0.0.0.0 \
    --server_port=8102 \
    --config_task_path=configs/task/robocasa.py
