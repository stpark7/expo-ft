#!/usr/bin/env bash
#
# RoboCasa365 (시뮬레이션) 정책 평가 — 체크포인트에서 롤아웃만(업데이트/버퍼 없음).
#
# 클라이언트(scripts/sim/run_policy.sh, client/sim/.venv)가 8102에서 listen 중이어야
# 한다. eval_droid_policy.py가 env_type='sim'을 받아 robocasa 경로로 동작한다
# (dataset에서 arm 7-dim example_action 도출, max_steps로 에피소드 상한).
#
# --checkpoint_dir : run_server.sh의 output_dir/run_name/checkpoints
#                    (= ./checkpoints/robocasa_sim/expo_robocasa_pickstove/checkpoints).
# --checkpoint_step 생략 시 최신 스텝 사용. pi05_* 값은 run_server.sh와 동일.

source .venv/bin/activate
CLIENT_IP=localhost

export CUDA_VISIBLE_DEVICES=0

CKPT=./checkpoints/robocasa365/pi05_pretrain_human300/multitask_learning/75000

python eval_droid_policy.py \
    --config_task=configs/task/robocasa.py \
    --config=configs/model/expo_ft_pi_config.py \
    --dataset_path=./data/robocasa365/v1.0/pretrain/atomic/PickPlaceCounterToStove/20250819/lerobot \
    --num_data=1 \
    --client_host="$CLIENT_IP" \
    --client_port=8102 \
    --config.N=8 \
    --config.n_edit_samples=8 \
    --config.edit_scale=0.2 \
    --config.pi05_config_name=pi05_pretrain_human300 \
    --config.pi05_weight_loader_path="$CKPT/params" \
    --config.pi05_assets_dir="$CKPT" \
    --config.pi05_asset_id="assets" \
    --checkpoint_dir=./checkpoints/robocasa_sim/expo_robocasa_pickstove/checkpoints \
    --num_episodes=25
