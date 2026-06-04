#!/usr/bin/env bash
#
# RoboCasa365 (시뮬레이션) RL learner(server)를 띄운다.
#
# RL-from-pretrained: SFT 단계 없이 사전학습 멀티태스크 체크포인트
# (pi05_pretrain_human300 / 75000)에서 바로 온라인 RL을 시작한다. 클라이언트
# (scripts/sim/run_policy.sh, client/sim/.venv)가 8102에서 listen하고 이 서버가
# 접속해 나온다. 다른 머신이면 scripts/set_server.sh로 리버스 터널을 먼저 연다.
#
# 경로/플래그는 손으로 배선한다(레지스트리 없음). robocasa 전용 핵심 값:
#   * config_task         : configs/task/robocasa.py (split=pretrain, residual 7-dim)
#   * pi05_config_name    : pi05_pretrain_human300 (openpi fork의 robocasa VLA config)
#   * pi05_weight_loader  : <ckpt>/75000/params (Orbax 사전학습 가중치 = RL 초기값)
#   * pi05_assets_dir/id  : <ckpt>/75000 + "assets" (norm_stats.json 위치; mean/std)
#   * offline_ratio=0.25  : 보호된 offline 데모 버퍼(RL-from-pretrained 초기 안정화)
#
# 새 run이면 resume/overwrite 없이 그대로 실행한다. 이어하려면 --resume,
# 같은 output_dir을 비우고 재시작하려면 --overwrite 를 추가한다.

source .venv/bin/activate

# SSH 리버스 터널이 8102를 localhost로 포워딩하므로 localhost로 접속.
CLIENT_IP=localhost

# 사용할 GPU. 동기 train_pi_robo.py는 단일 mesh라 GPU 1장으로 동작한다.
# (GPU가 여러 장이면 0,1,2,3 식으로 늘리고 --fsdp_devices도 맞춰서 키운다.)
export CUDA_VISIBLE_DEVICES=0

# 사전학습 체크포인트 루트(다운로드된 Orbax 체크포인트).
CKPT=./checkpoints/robocasa365/pi05_pretrain_human300/multitask_learning/75000

python train_pi_robo.py \
    --config_task=configs/task/robocasa.py \
    --dataset_path=./data/robocasa365/v1.0/pretrain/atomic/PickPlaceCounterToStove/20250819/lerobot \
    --num_data=50 \
    --update_type=episode \
    --num_updates=3 \
    --offline_ratio=0.25 \
    --config=configs/model/expo_ft_pi_config.py \
    --config.N=8 \
    --config.n_edit_samples=8 \
    --config.edit_scale=0.2 \
    --config.pi05_config_name=pi05_pretrain_human300 \
    --config.pi05_weight_loader_path="$CKPT/params" \
    --config.pi05_assets_dir="$CKPT" \
    --config.pi05_asset_id="assets" \
    --project_name=expo_ft_robocasa \
    --output_dir=./checkpoints/robocasa_sim \
    --client_host="$CLIENT_IP" \
    --client_port=8102 \
    --fsdp_devices=1 \
    --checkpoint_model \
    --checkpoint_buffer \
    --checkpoint_interval=2000 \
    --resume \
    --run_name=expo_robocasa_pickstove
