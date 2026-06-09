#!/usr/bin/env bash

source .venv/bin/activate

# SSH 리버스 터널이 8102를 localhost로 포워딩하므로 localhost로 접속.
CLIENT_IP=localhost

# 사용할 GPU. 동기 train_pi_robo.py는 단일 mesh라 GPU 1장으로 동작한다.
# (GPU가 여러 장이면 0,1,2,3 식으로 늘리고 --fsdp_devices도 맞춰서 키운다.)
export CUDA_VISIBLE_DEVICES=0

# JAX가 GPU에서 선점할 VRAM 비율(0.9=전체의 90%만 미리 확보)
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

# XLA GPU 커널 자동 튜닝 수준(3=표준): 여러 커널 구현을 벤치마크해 가장 빠른 것을 선택
export XLA_FLAGS="--xla_gpu_autotune_level=3"

# 사전학습 체크포인트 루트(다운로드된 Orbax 체크포인트).
CKPT=./checkpoints/robocasa365/pi05_pretrain_human300/multitask_learning/75000

python train_pi_robo.py \
    --config_task=configs/task/robocasa.py \
    --dataset_path=./data/robocasa365/v1.0/pretrain/atomic/PickPlaceCounterToStove/20250819/lerobot \
    --num_data=100 \
    --buffer_capacity=30000 \
    `# ── 업데이트 스케줄 ─────────────────────────────────────────────────────` \
    `# update_type: 언제 그래디언트 업데이트를 돌릴지. episode=에피소드 끝마다(현재),` \
    `#   step=스텝마다, batch=num_batch개 에피소드 모아서.` \
    --update_type=episode \
    `# num_updates: 트리거 1회당 업데이트 횟수(에피소드당 3회).` \
    --num_updates=3 \
    `# offline_ratio: critic/actor 배치에 섞을 offline 데모 비율. 0.25 = 보호된 offline` \
    `#   데모 버퍼 25% 혼합(RL-from-pretrained 초기 안정화). 0이면 데모를 online 버퍼에 시드.` \
    `#   [진단] 평평하게 안 오르면 offline 데모 간섭 의심 → 0으로 빼고 재실행해 알고리즘과 분리.` \
    --offline_ratio=0.25 \
    `# truncate_offline_at_success: 데모를 첫 성공에서 잘라 done=1(단일-terminal 보상).` \
    `#   온라인 env가 첫 성공에 종료하는 것과 n-step critic 타깃을 일치(둘 다 ≈1.0)시킨다.` \
    `#   끄면(False) 데모의 held-reward(16프레임)로 offline 타깃이 ≈7.7로 부풀어 Q가 과대.` \
    `#   [A/B] 이 플래그 True/False로 같은 seed 비교하면 효과 격리 가능.` \
    --truncate_offline_at_success=True \
    `# utd_ratio: Update-to-Data. 한 배치를 미니배치로 쪼개 critic을 몇 번 스캔할지.` \
    --utd_ratio=20 \
    `# batch_size: 미니배치 크기(device 수로 나누어떨어져야 함). 2 = 32GB 단일 GPU OOM 회피값.` \
    --batch_size=24 \
    `# ── 모델/알고리즘 config (expo_ft_pi_config.py, EXPOLearner) ─────────────` \
    --config=configs/model/expo_ft_pi_config.py \
    `# N: pi0.5 base가 뽑는 후보 action chunk 수(config 기본 8). 2 = OOM 회피용 축소.` \
    `#   [진단] 후보 다양성을 늘려보려면 4까지 올렸었다(아래 edit_scale/num_qs와 함께 peak 상승 주의).` \
    --config.N=8 \
    `# n_edit_samples: residual actor가 편집할 후보 수(보통 N과 동일하게). 2(기본)/4(진단).` \
    --config.n_edit_samples=8 \
    `# edit_scale: residual 편집 크기 스케일. 0.2=보수적 기본. [진단] 더 크게 밀어보려면 0.5.` \
    --config.edit_scale=0.2 \
    `# num_qs: Q-앙상블 개수(config 기본 10). OOM 대책으로 3까지 낮춤 — 정상 사용량이 이미` \
    `#   ~30.8GB/32GB라 freeze_pi05_encoder만으론 부족. Q-앙상블이 ResNet×3카메라로 _update_jit` \
    `#   peak의 핵심이다. 첫 update(~step3300) 컴파일 그래프(~19.3GB)가 MEM_FRACTION 0.7 예산` \
    `#   (22.8GB)을 초과하면 XLA rematerialization→내부 CHECK abort(shape_util.cc, "IsTuple")로` \
    `#   죽는다. N/n_edit과 함께 update peak를 remat 임계 아래로 낮춰 abort 회피.` \
    --config.num_qs=10 \
    `# freeze_pi05_actor: True면 action expert backward까지 스킵 → pi0.5 activation 제거로` \
    `#   메모리 절감. critic + residual actor + temperature만 RL로 학습(RL-from-pretrained와 일관).` \
    `#   vision encoder는 config의 freeze_pi05_encoder=True로 이미 동결돼 있다.` \
    --config.freeze_pi05_actor=True \
    `# ── 사전학습 가중치/에셋 (절대경로, 손으로 배선) ─────────────────────────` \
    `# pi05_config_name: openpi fork의 VLA config 이름(=VLA 구조의 source of truth).` \
    `#   _lora 변형 → base 얼리고 LoRA 어댑터만 학습 → train state ≈13 GiB(32GB GPU 적합).` \
    --config.pi05_config_name=pi05_pretrain_human300_lora \
    `# pi05_weight_loader_path: Orbax 사전학습 가중치(=LoRA base 초기값). 빠진 LoRA 어댑터는 새로 초기화.` \
    --config.pi05_weight_loader_path="$CKPT/params" \
    `# pi05_assets_dir + pi05_asset_id: norm_stats.json 위치(<ckpt>/assets; mean/std).` \
    --config.pi05_assets_dir="$CKPT" \
    --config.pi05_asset_id="assets" \
    `# ── 환경 시드 (모드 전환 핵심 플래그) ───────────────────────────────────` \
    `# fix_env_seed: ≥0이면 매 reset마다 그 seed로 sim 장면 전체(주방+객체+배치+언어)를 고정` \
    `#   (단일 환경 학습/완전 재현). <0(-1)이면 무작위.` \
    `#   [진단/주방고정] 여기서는 -1을 쓴다. 주방만 고정하는 것은 클라이언트 run_policy.sh의` \
    `#   --fixed_layout_id/--fixed_style_id가 담당하므로, 서버에서 seed까지 고정하면(>=0)` \
    `#   객체/배치/언어까지 전부 같은 장면으로 묶여 '주방만 고정' 의도와 정반대가 된다.` \
    --fix_env_seed=-1 \
    `# ── 분산/접속 ───────────────────────────────────────────────────────────` \
    --project_name=expo_ft_robocasa \
    --output_dir=./log/robocasa_sim \
    `# client_host/port: 리버스 터널로 localhost:8102에 접속(클라가 거기 listen).` \
    --client_host="$CLIENT_IP" \
    --client_port=8102 \
    `# fsdp_devices: FSDP 샤딩 디바이스 수. 단일 GPU라 1.` \
    --fsdp_devices=1 \
    `# ── 체크포인트 ──────────────────────────────────────────────────────────` \
    `# checkpoint_model: 에이전트 가중치 저장. checkpoint_buffer: 리플레이 transition까지` \
    `#   저장(정확한 resume용). checkpoint_interval: 이 스텝마다 저장.` \
    --checkpoint_model \
    --checkpoint_buffer \
    --checkpoint_interval=10000 \
    `# ── 학습 중 평가(in-training eval) ────────────────────────────────────` \
    `# eval_interval env 스텝마다 에피소드 경계에서 잠시 멈추고 결정적 seed로 평가 에피소드를` \
    `# 굴려 성공률/리턴/길이를 콘솔(Eval at step N: ...)과 wandb(eval/*)에 찍는다. 끝까지` \
    `# 기다리지 않고 RL이 잘 되는지 중간에 본다. 평가는 실시간 페이싱이라 비용이 크니` \
    `# (sim 20Hz·max_steps400 → 최대 ~20s/ep) eval_episodes는 작게 둔다.` \
    `# (주방고정 진단 땐 통계 안정용으로 20까지 올렸었다.)` \
    `# eval_seed: 평가 reset seed의 base(sim: seed+ep). 매 평가 동일 장면 세트 재현용.` \
    `# eval_base_at_start: 시작 시 pi0.5 base만 1회 평가 → RL(eval/*)이 base(eval_base/*)를` \
    `# 넘는지 비교하는 step0 기준선. (full 정책 step0 평가는 critic이 랜덤이라 무의미해 제거됨.)` \
    --eval_interval=10000 \
    --eval_episodes=20 \
    --eval_seed=42 \
    --eval_base_at_start \
    `# eval_base_at_start: 활성화 — step0에 base 기준선(eval_base/*)을 찍어 RL(eval/*)과 비교.` \
    `# ── 재개/덮어쓰기 + run 이름 ────────────────────────────────────────────` \
    `# resume: 같은 output_dir/run_name 체크포인트에서 이어하기(현재).` \
    `# overwrite: 같은 디렉터리를 비우고 처음부터(새 진단 run으로 버퍼 오염 방지할 때).` \
    `#   → 둘 중 하나만 켠다. 깨끗한 새 run이면 --resume을 --overwrite로 바꾼다.` \
    --overwrite \
    `# run_name: 로그/체크포인트 디렉터리 이름. 진단처럼 버퍼를 분리하려면 새 이름 + --overwrite.` \
    `#   (주방고정 진단 예전 이름: expo_robocasa_pickstove_fixedkitchen_L11S14)` \
    --run_name=expo_robocasa_pickstove_armfix_scale0.2


# 변수 수정 가이드
# buffer_size : 서버에서 돌릴 시 조금 더 키워야함.
# 