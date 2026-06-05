#!/usr/bin/env bash
#
# RoboCasa365 (시뮬레이션) RL learner(server)를 띄운다.
#
# RL-from-pretrained: SFT 단계 없이 사전학습 멀티태스크 체크포인트
# (pi05_pretrain_human300 / 75000)에서 바로 온라인 RL을 시작한다. 단, 풀 파인튜닝은
# train state ≈50 GiB라 OOM이므로 LoRA 변형(pi05_pretrain_human300_lora)으로 돌린다
# → train state ≈13 GiB라 RTX 5090 한 장에 들어간다. 클라이언트
# (scripts/sim/run_policy.sh, client/sim/.venv)가 8102에서 listen하고 이 서버가
# 접속해 나온다. 다른 머신이면 scripts/set_server.sh로 리버스 터널을 먼저 연다.
#
# 경로/플래그는 손으로 배선한다(레지스트리 없음). robocasa 전용 핵심 값:
#   * config_task         : configs/task/robocasa.py (split=pretrain, residual 7-dim)
#   * pi05_config_name    : pi05_pretrain_human300_lora (openpi fork의 robocasa LoRA VLA
#                           config; base 얼리고 LoRA 어댑터만 학습 → train state ≈13 GiB)
#   * pi05_weight_loader  : <ckpt>/75000/params (Orbax 사전학습 가중치 = LoRA base 초기값.
#                           빠진 LoRA 어댑터는 새로 초기화됨)
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

# JAX/XLA 메모리: 기본값은 GPU의 75%만 선점(24GB→~18GB 상한)이라 OOM 여유가 적다.
# 32GB GPU(현재 머신)에서 LoRA train state(~13GiB) + 데이터 프리페치 이미지를 돌리되,
# 0.95는 step 2000 체크포인트(Orbax) 직렬화 시 호스트/디바이스 메모리 여유가 부족해
# 프로세스가 죽는 의심이 있어 0.8로 낮췄었다(약 26GB 상한).
# 그런데 rustdesk/gnome-remote-desktop이 GPU를 ~5.8GB 점유한 상태에선 0.8(≈26GB) 선점이
# 남는 여유를 ~0.8GB까지 깎아 startup 첫 GEMM에서 cuBLAS 초기화 실패
# ("Failed to initialize BLAS support for GemmCmd")로 죽는다. → 0.7로 낮춰 cuBLAS/오토튜너용
# 헤드룸(~4GB) 확보. 원격데스크톱을 끄면 0.8로 올려도 된다.
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7

# step ~3200 그래디언트 업데이트(_update_jit) 중 GPU OOM: 정상 사용량(~30.8GB)은 32GB에
# 들어가지만 XLA 오토튜너가 GEMM 커널 벤치마킹에 임시 ~1.9GB를 더 잡으려다 터졌다.
# (에러 메시지 권고) autotune level=3 → 정확성 검사용 레퍼런스 버퍼를 생략해 오토튜너
# 메모리를 줄인다(커널 선택 자체는 유지, 학습 정확도 영향 없음). 그래도 또 OOM이면
# batch_size / num_qs / N 같은 워크로드 자체를 줄여야 한다.
export XLA_FLAGS="--xla_gpu_autotune_level=3"

# 사전학습 체크포인트 루트(다운로드된 Orbax 체크포인트).
CKPT=./checkpoints/robocasa365/pi05_pretrain_human300/multitask_learning/75000

python train_pi_robo.py \
    --config_task=configs/task/robocasa.py \
    --dataset_path=./data/robocasa365/v1.0/pretrain/atomic/PickPlaceCounterToStove/20250819/lerobot \
    --num_data=50 \
    `# 리플레이 버퍼 capacity를 max_steps(100k)에서 분리해 고정 크기 원형 버퍼로 묶는다.` \
    `# 호스트 RAM 근본 대책: 버퍼가 max_steps까지 자라면(~58GB) 체크포인트 때 params Orbax` \
    `# 직렬화 호스트 복사본(~13GB)이 얹혀 62GB 머신에서 커널 global_oom으로 죽었다(2026-06-05 18:05,` \
    `# anon-rss 50.3GiB). 40000이면 버퍼 ≈24GB로 고정 → 체크포인트 peak에도 여유. resume 시` \
    `# 저장된 ~47k transition 중 최근 40k만 원형으로 복원(가장 오래된 ~7k drop, 리플레이라 무해).` \
    --buffer_capacity=40000 \
    --update_type=episode \
    --num_updates=3 \
    --offline_ratio=0.25 \
    --utd_ratio=2 \
    --batch_size=2 \
    --config=configs/model/expo_ft_pi_config.py \
    --config.N=2 \
    --config.n_edit_samples=2 \
    --config.edit_scale=0.2 \
    `# OOM 대책: 정상 사용량이 이미 ~30.8GB/32GB라 freeze_pi05_encoder만으론 부족하다.` \
    `# (1) num_qs 10→3: Q-앙상블이 ResNet×3카메라로 _update_jit peak의 핵심.` \
    `#     첫 update(~step3300) 컴파일 그래프(~19.3GB)가 MEM_FRACTION 0.7 예산(22.8GB)을 초과해`\
    `#     XLA rematerialization→내부 CHECK abort(shape_util.cc, "Check failed: IsTuple")로 죽었다.`\
    `#     N 4→2, n_edit 4→2와 함께 update peak를 remat 임계 아래로 낮춰 abort 회피.` \
    --config.num_qs=3 \
    `# (2) action expert까지 진짜 동결: update_actor backward 자체를 스킵해 pi0.5 activation 제거.` \
    `#     (critic + residual actor + temperature만 RL로 학습. RL-from-pretrained 전략과 일관)` \
    --config.freeze_pi05_actor=True \
    --config.pi05_config_name=pi05_pretrain_human300_lora \
    --config.pi05_weight_loader_path="$CKPT/params" \
    --config.pi05_assets_dir="$CKPT" \
    --config.pi05_asset_id="assets" \
    --project_name=expo_ft_robocasa \
    --output_dir=./log/robocasa_sim \
    --client_host="$CLIENT_IP" \
    --client_port=8102 \
    --fsdp_devices=1 \
    --checkpoint_model \
    --checkpoint_buffer \
    --checkpoint_interval=10000 \
    --resume \
    --run_name=expo_robocasa_pickstove_armfix
