#!/usr/bin/env bash
#
# RoboCasa365 (시뮬레이션) 정책 평가 — 체크포인트에서 롤아웃만(업데이트/버퍼 없음).
#
# 사용법:
#   scripts/sim/run_policy.sh        # (다른 터미널) sim 클라이언트를 먼저 8102에 띄운다
#   scripts/sim/eval_policy.sh full  # 학습된 EXPO 정책(residual+critic) 50 에피소드
#   scripts/sim/eval_policy.sh base  # pi0.5 base 정책만(residual/critic 없음) 50 에피소드
#
# 두 모드 모두 같은 체크포인트(STEP)를 복원하고 같은 --seed로 돌리므로, 50개 장면이
# 양쪽 동일하다(sim은 reset_seed=seed+ep로 고정). 따라서 full vs base 성공률을 그대로
# paired 비교할 수 있다. 학습 때 pi0.5 actor가 freeze_pi05_actor=True로 동결됐으므로
# base 모드 결과가 곧 사전학습(75000) base 정책의 성능이다.
#
# ⚠️ 평가 config는 학습(run_server.sh)과 네트워크 구조가 일치해야 복원이 된다. 특히:
#   * --config.num_qs=3  : critic 앙상블 크기. 학습이 3이라 여기서도 3이어야 한다
#                          (config 기본값 10으로 두면 Orbax 복원이 shape 불일치로 깨진다).
#   * pi05_config_name=pi05_pretrain_human300_lora : LoRA 어댑터 포함 모델 구조.
#   * pi05_weight_loader/assets : LoRA base 초기값 + norm_stats(mean/std) 위치(=75000).

set -euo pipefail

source .venv/bin/activate
CLIENT_IP=localhost
export CUDA_VISIBLE_DEVICES=0

# ── 모드 인자: full(기본) = 학습된 정책, base = pi0.5 base만 ──────────────
MODE="${1:-full}"
case "$MODE" in
    full) ONLY_BASE=false ;;
    base) ONLY_BASE=true  ;;
    *) echo "usage: $0 [full|base]"; exit 1 ;;
esac

# 사전학습 체크포인트(= LoRA base 초기값 + norm_stats). run_server.sh와 동일.
CKPT=./checkpoints/robocasa365/pi05_pretrain_human300/multitask_learning/75000

# 평가할 RL 체크포인트(run_server.sh의 output_dir/run_name/checkpoints) + step.
RL_CKPT_DIR=./log/robocasa_sim/expo_robocasa_pickstove_armfix/checkpoints
# STEP / N_EPISODES 는 환경변수로 덮어쓸 수 있다(scripts/sim/eval_both.sh가 그렇게 호출한다).
# 단독 실행 시 기본값은 step 100000 / 100 에피소드.
STEP="${STEP:-100000}"
N_EPISODES="${N_EPISODES:-100}"

echo ">>> eval mode=$MODE  step=$STEP  episodes=$N_EPISODES  only_base_actions=$ONLY_BASE"

python eval_droid_policy.py \
    --config_task=configs/task/robocasa.py \
    --config=configs/model/expo_ft_pi_config.py \
    --dataset_path=./data/robocasa365/v1.0/pretrain/atomic/PickPlaceCounterToStove/20250819/lerobot \
    --num_data=1 \
    --client_host="$CLIENT_IP" \
    --client_port=8102 \
    --seed=42 \
    --config.N=8 \
    --config.n_edit_samples=8 \
    --config.edit_scale=0.2 \
    --config.num_qs=3 \
    --config.pi05_config_name=pi05_pretrain_human300_lora \
    --config.pi05_weight_loader_path="$CKPT/params" \
    --config.pi05_assets_dir="$CKPT" \
    --config.pi05_asset_id="assets" \
    --checkpoint_dir="$RL_CKPT_DIR" \
    --checkpoint_step="$STEP" \
    --only_base_actions="$ONLY_BASE" \
    --num_episodes="$N_EPISODES"
