#!/usr/bin/env python
"""test_vram_train.py — '실제 학습 한 스텝'이 정적 상주 VRAM 위로 얼마나, 왜 더 먹는지 실측.

test_vram.py 와의 관계
----------------------
- test_vram.py     : 신경망 가중치/옵티마이저의 **정적 상주(resident)** VRAM 만 잰다.
                     (모델을 GPU에 올려두기만 했을 때의 bytes_in_use)
- test_vram_train.py(이 파일): 그 위에서 **agent.update(=학습 한 스텝)** 를 실제로 돌렸을 때
                     순간적으로 치솟는 **peak** 을 잰다. 학습 중 OOM 을 만드는 건 이 peak 이다.

왜 학습할 때 VRAM 이 '또' 올라가나 (개념) — 4개 버킷
----------------------------------------------------
학습 1스텝 VRAM 은 정확히 4개로 쪼개진다(파라미터 N개, dtype별 바이트):
  ① 파라미터(weights)      : 모델 가중치 자체. **상주**(batch 무관). pi0.5 base=bf16(2B),
                             critic/encoder/residual/temp 와 LoRA=fp32(4B).
  ② 그래디언트(gradients)  : '학습 대상' 파라미터 크기만큼 1벌. backward 동안만(=transient).
                             freeze_pi05_actor=True 면 pi0.5 는 grad 0 → critic+encoder+
                             residual_actor+temperature(전부 fp32)만 grad 를 만든다.
  ③ 옵티마이저(Adam m,v)   : 학습 대상마다 모멘트 2벌(fp32). **상주**. pi0.5 는 set_to_zero 라 ~0.
  ④ 액티베이션(activations): forward 중간값을 backward 용으로 보존. **batch 에 비례**(transient).
                             + n-step 타깃·이미지 증강 중간버퍼·XLA fusion/컴파일 scratch.

①③ 은 batch 와 **무관**(고정), ④ 만 batch·해상도와 함께 출렁인다 — batch 를 줄이면 OOM 이
풀리는 이유가 바로 ④ 다. EXPO 특이점: utd_ratio 가 한 배치를 미니배치로 쪼개 lax.scan 하므로
보존 액티베이션은 '미니배치(batch_size)'에 비례하고, 입력 이미지배치(arg)만 '총배치
(batch_size×utd_ratio)'에 비례한다(=scan 이 gradient-accumulation 처럼 동작).

이 스크립트는 ①②③ 을 빌드된 네트워크 트리에서 **정확히** 계산(이론)하고, ④ 는 추정한 뒤,
compiled memory_analysis 의 실측값과 **나란히 비교**한다.

EXPO 의 update 한 스텝(_update_jit)은 특히 아래 셋이 activation 을 만든다:
  (1) sample_batch_actions : pi0.5 base 가 후보 (N + n_edit_samples)개를 forward
                             — critic 미니배치마다 반복(utd_ratio 회)
  (2) critic               : ResNetV2 인코더 × 카메라 3대(9채널) × Q앙상블(num_qs) 의
                             forward + backward (+ 인코더도 학습이면 backward)
  (3) update_actor         : freeze_pi05_actor=False 일 때만. pi0.5 action expert 의
                             forward + backward — 단일 activation 덩어리로는 보통 최대

이 activation 들은 jit 함수가 끝나면 해제돼 bytes_in_use 는 상주 R 로 되돌아온다. 하지만
그 사이의 **peak_bytes_in_use 가 GPU 에 실제로 맞아야 하는 값**이고, run_server.sh 주석이
말하는 "첫 update 컴파일 그래프(~19.3GB)가 예산 초과 → remat → abort" 가 바로 이 peak 다.
그래서 정적 R 이 여유 있어 보여도 update peak 에서 터진다.

측정 방법 (왜 서브프로세스로 쪼개나)
------------------------------------
peak_bytes_in_use 는 **프로세스 수명 동안의 누적 최댓값**이라 리셋이 안 된다. 같은
프로세스에서 update 를 여러 변형으로 재면 peak 이 섞여 컴포넌트별 분해가 불가능하다.
그래서 각 변형을 **독립 서브프로세스**로 돌려 깨끗한 peak 을 얻고, 드라이버가 순차 실행
(서로 GPU 를 안 뺏게)한 뒤 결과를 모아 표로 출력한다.

컴포넌트 분해는 '실제 agent.update' 를 config 만 바꿔 돌려서 한다(가짜 경로가 아님):
  · critic-only   : freeze_pi05_actor=True, n_edit_samples=0  → (1)+(2) 만
  · REAL(런과 동일): freeze_pi05_actor=True, n_edit_samples=4  → +residual actor+temperature
  · unfreeze actor: freeze_pi05_actor=False, n_edit_samples=4 → +(3) pi0.5 actor backward
  차이가 곧 각 컴포넌트의 spike 기여분.

기본값 = scripts/sim/run_server.sh 의 robocasa 실제 런과 동일 (★ 이 값으로 이론/실측을 낸다)
  num_qs=10, N=8, n_edit_samples=8, edit_scale=0.5, batch_size=32, utd_ratio=20, replan_steps=8,
  freeze_pi05_actor=True, critic 3카메라(9채널), pi05_config=pi05_pretrain_human300_lora.
  → critic 배치 = batch_size×utd_ratio = 640 (미니배치 32 × scan 20회).

실행
----
  source .venv/bin/activate

  # (A) 체크포인트 없이 — 랜덤 init(메모리는 가중치 '값'과 무관해 실제와 동일). 어디서나 실행 가능:
  CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
      python test_vram_train.py --random-init

  # (B) 실제 런과 동일 — 체크포인트/norm_stats 로드(rollout sample 도 측정):
  CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
      python test_vram_train.py --ckpt ./checkpoints/robocasa365/pi05_pretrain_human300/multitask_learning/75000

  # (C) 컴포넌트 분해(critic-only / REAL / unfreeze + rollout) — 서브프로세스 4~5개 순차:
  python test_vram_train.py --random-init --breakdown

주요 옵션
  --random-init / --ckpt PATH         : 가중치 소스(랜덤 vs 실제 체크포인트)
  --breakdown                         : 컴포넌트별 분해 표(여러 서브프로세스)
  --unfreeze-actor                    : 단일 측정 시 pi0.5 base actor 까지 학습(무거운 경우)
  --batch-size/--utd-ratio/--N/--n-edit-samples/--num-qs/--replan-steps/--resize
  --pi05-config-name/--fsdp-devices/--seed
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile


# ──────────────────────────────────────────────────────────────────────────
# 단위 변환
# ──────────────────────────────────────────────────────────────────────────
def mib(b):
    return b / (1024 ** 2)


def gib(b):
    return b / (1024 ** 3)


# ──────────────────────────────────────────────────────────────────────────
# GPU 메모리 통계 (worker 안에서만 사용 — jax 필요)
# ──────────────────────────────────────────────────────────────────────────
def _stat(key):
    import jax
    try:
        stats = jax.local_devices()[0].memory_stats() or {}
    except Exception:
        return -1
    return int(stats.get(key, -1))


def gpu_in_use():
    """현재 GPU 가 실제로 잡고 있는 live buffer 바이트. CPU/미지원이면 -1."""
    return _stat("bytes_in_use")


def gpu_peak():
    """프로세스 수명 동안의 누적 peak. (리셋 불가 → 변형별로 서브프로세스 분리)"""
    return _stat("peak_bytes_in_use")


def gpu_limit():
    """MEM_FRACTION 이 반영된 GPU 메모리 예산."""
    return _stat("bytes_limit")


# ──────────────────────────────────────────────────────────────────────────
# 에이전트 빌드 (train_pi_robo.py 의 build_pi05 + load_agent 경로를 그대로 재현)
# ──────────────────────────────────────────────────────────────────────────
def _build_pi05_random(config, seed, mesh, data_sharding, replicated_sharding, default_prompt):
    """체크포인트 없이 pi0.5 를 랜덤 init 으로 빌드.

    build_pi05() 와 동일하되 weight_loader 만 NoOpWeightLoader 로 바꿔 디스크 가중치 로드를
    건너뛴다. NoOp 은 입력 shape 를 그대로 돌려주므로 _load_weights_and_validate 가
    빈 partial_params({})를 만들고 model 은 랜덤 init 상태로 남는다. 메모리 footprint(=shape/
    dtype)는 실제 체크포인트와 100% 동일하다(값만 랜덤). norm_stats 가 없어도 data.create 는
    FileNotFoundError 를 잡아 None 을 돌려주므로 빌드는 통과한다(rollout sample 만 못 씀).
    """
    import dataclasses
    import jax
    from expo_ft.utils.train_utils import build_pi05_config
    from expo_ft.agents.vla.pi05 import Pi05Agent
    from openpi.training.weight_loaders import NoOpWeightLoader

    agent_kwargs, pi05_train_config, pi05_resize_size, _ = build_pi05_config(config)
    pi05_train_config = dataclasses.replace(pi05_train_config, weight_loader=NoOpWeightLoader())

    freeze_encoder = agent_kwargs.pop("freeze_pi05_encoder", False)
    freeze_actor = bool(agent_kwargs.get("freeze_pi05_actor", False))

    rng = jax.random.PRNGKey(seed)
    init_rng, rng = jax.random.split(rng)
    target_rng, rng = jax.random.split(rng)

    actor, actor_train_state, _ = Pi05Agent.initialize(
        pi05_train_config, mesh, init_rng,
        resume=False, default_prompt=default_prompt,
        data_sharding=data_sharding, replicated_sharding=replicated_sharding,
        freeze_pi05_encoder=freeze_encoder, infer_device=jax.devices()[0],
        freeze_actor=freeze_actor,
    )
    if freeze_actor:
        # frozen 이면 target 은 update_actor(스킵)에서만 읽히므로 참조 공유로 충분(추가 VRAM 0).
        target_actor_params = actor.get_params(actor_train_state)
    else:
        target_actor_params = actor.init_target_params(target_rng, resume=False)

    metadata = dict(
        action_horizon=pi05_train_config.model.action_horizon,
        resize_size=pi05_resize_size,
        freeze_encoder=freeze_encoder,
    )
    return actor, actor_train_state, target_actor_params, agent_kwargs, metadata


def build_agent(args):
    """expo_ft 모델/태스크 config 를 run_server.sh 와 동일하게 세팅하고 EXPO 에이전트를 만든다."""
    import jax
    import openpi.training.sharding as openpi_sharding

    from configs.model import expo_ft_pi_config
    from configs.task import robocasa
    from expo_ft.agents.alg.expo_ft import load_agent

    # ── 모델 config (+ run_server.sh 오버라이드) ──────────────────────────
    config = expo_ft_pi_config.get_config()
    config.unlock()
    config.num_qs = args.num_qs
    config.N = args.N
    config.n_edit_samples = args.n_edit_samples
    config.edit_scale = args.edit_scale
    config.freeze_pi05_actor = bool(args.freeze_actor)
    config.pi05_config_name = args.pi05_config_name
    if args.random_init:
        config.pi05_weight_loader_path = ""
        config.pi05_assets_dir = ""
        config.pi05_asset_id = ""
    else:
        config.pi05_weight_loader_path = f"{args.ckpt}/params"
        config.pi05_assets_dir = args.ckpt
        config.pi05_asset_id = "assets"

    # ── 태스크 config (robocasa) ──────────────────────────────────────────
    config_task = robocasa.get_config()
    config_task.unlock()
    # language_instruction 은 placeholder 라 접근 전에 채워야 함(메모리와 무관, 빌드용 더미).
    config_task.language_instruction = "pick up the object and place it on the stove"
    task_description = config_task.language_instruction
    critic_camera_keys = tuple(config_task.critic_camera_keys)

    # 이론(④ 액티베이션 추정)용으로 모델 config 의 차원들을 args 에 실어둔다.
    args.encoder_stage_sizes = tuple(config.encoder_stage_sizes)
    args.encoder_num_filters = int(config.encoder_num_filters)
    args.hidden_dims = tuple(config.hidden_dims)
    args.latent_dim_image = int(config.latent_dim_image)
    args.latent_dim_state = int(config.latent_dim_state)

    # ── 샤딩 (train_pi_robo.py 와 동일) ───────────────────────────────────
    mesh = openpi_sharding.make_mesh(args.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(openpi_sharding.DATA_AXIS)
    )
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # ── pi0.5 빌드 (실제 체크포인트 or 랜덤 init) ────────────────────────
    # 체크포인트 경로가 없으면 자동으로 랜덤 init 로 폴백(메모리는 동일).
    if (not args.random_init) and (not os.path.isdir(f"{args.ckpt}/params")):
        print(f"[경고] 체크포인트 없음: {args.ckpt}/params → 랜덤 init 로 폴백(메모리는 동일).")
        args.random_init = True
        config.pi05_weight_loader_path = ""
        config.pi05_assets_dir = ""
        config.pi05_asset_id = ""
    if args.random_init:
        actor, ats, target, agent_kwargs, metadata = _build_pi05_random(
            config, args.seed, mesh, data_sharding, replicated_sharding, task_description
        )
    else:
        from expo_ft.agents.vla.pi05 import build_pi05
        actor, ats, target, agent_kwargs, metadata = build_pi05(
            config, args.seed, mesh, data_sharding, replicated_sharding,
            False, task_description,
        )

    # ── 예시 shape (replay buffer 없이 직접 합성: env/데이터셋 불필요) ────
    # critic 입력은 카메라 3대를 채널방향 concat → [H, W, 9]. arm action 7, state 16.
    action_dim = 7
    state_dim = 16
    num_cams = len(critic_camera_keys)
    H = W = args.resize
    import numpy as np
    example_observation = np.zeros((H, W, 3 * num_cams), dtype=np.float32)
    example_action = np.zeros((metadata["action_horizon"], action_dim), dtype=np.float32)
    example_state = np.zeros((state_dim,), dtype=np.float32)

    actor.action_dim = action_dim
    actor.state_dim = state_dim
    actor.action_pad_offset = config_task.get("action_pad_offset", 0)

    agent = load_agent(
        seed=args.seed,
        example_observation=example_observation,
        example_action=example_action,
        example_state=example_state,
        actor=actor,
        actor_train_state=ats,
        target_actor_params=target,
        agent_kwargs=agent_kwargs,
        metadata=metadata,
        mesh=mesh,
        data_sharding=data_sharding,
        replicated_sharding=replicated_sharding,
        resume=False,
        replan_steps=args.replan_steps,
        default_prompt=task_description,
        residual_action_xyzg=config_task.residual_action_xyzg,
        critic_camera_keys=critic_camera_keys,
    )
    return agent, data_sharding, replicated_sharding


# ──────────────────────────────────────────────────────────────────────────
# 합성 배치 (replay_buffer._convert_to_openpi_format 출력과 동일한 형식)
# ──────────────────────────────────────────────────────────────────────────
_CAMS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def _make_openpi_batch(agent, batch_size, resize, data_sharding):
    """update() 가 받는 그대로의 openpi-format 배치를 합성해 device 에 올린다.

    값은 의미 없음(메모리만 측정). shape/dtype/키 는 실제 학습 배치와 동일하게 맞춘다:
    이미지는 [-1,1] float32 [B,H,W,3], state/next_state 는 padded_dim(32), actions 는
    [B, action_horizon, padded_dim].
    """
    import numpy as np
    import jax

    padded_dim = agent.actor.model_config.action_dim          # 32 (모델 패딩 차원)
    action_horizon = agent.action_horizon
    max_token_len = agent.actor.model_config.max_token_len
    B, H, W = batch_size, resize, resize

    def img():
        return (np.random.rand(B, H, W, 3).astype(np.float32) * 2.0 - 1.0)

    def mask():
        return np.ones((B,), dtype=bool)

    batch = {
        "image": {k: img() for k in _CAMS},
        "image_mask": {k: mask() for k in _CAMS},
        "next_image": {k: img() for k in _CAMS},
        "next_image_mask": {k: mask() for k in _CAMS},
        "state": np.random.rand(B, padded_dim).astype(np.float32),
        "next_state": np.random.rand(B, padded_dim).astype(np.float32),
        "actions": np.random.rand(B, action_horizon, padded_dim).astype(np.float32),
        "tokenized_prompt": np.zeros((B, max_token_len), dtype=np.int32),
        "tokenized_prompt_mask": np.ones((B, max_token_len), dtype=bool),
        "rewards": np.random.rand(B).astype(np.float32),
        "masks": np.ones((B,), dtype=np.float32),
        "dones": np.zeros((B,), dtype=bool),
        "valids": np.ones((B,), dtype=np.float32),
        "is_hil": np.zeros((B,), dtype=bool),
        "hil_chunk": np.zeros((B,), dtype=bool),
        "is_success": np.ones((B,), dtype=bool),
    }
    # 실제 경로(apply_data_sharding)처럼 device 로 올려, 타이밍/peak 측정에서 host→device
    # 전송이 섞이지 않게 한다.
    return jax.tree_util.tree_map(lambda x: jax.device_put(x, data_sharding), batch)


def _tree_nbytes(tree):
    """pytree 안 array leaf 들의 총 바이트(이론값, shape×dtype)."""
    import jax
    import numpy as np
    total = 0
    for l in jax.tree_util.tree_leaves(tree):
        if hasattr(l, "shape") and hasattr(l, "dtype"):
            total += int(np.prod(l.shape)) * np.dtype(l.dtype).itemsize
    return total


# ──────────────────────────────────────────────────────────────────────────
# 표 정렬 유틸 — 한글(전각)은 폭 2로 세서 터미널에서 열이 진짜로 맞게 한다.
# ──────────────────────────────────────────────────────────────────────────
import unicodedata


def _dw(s: str) -> int:
    """문자열의 '터미널 표시 폭'. 전각(한글/CJK)은 2, 반각은 1."""
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in str(s))


def _lj(s, w):
    s = str(s)
    return s + " " * max(0, w - _dw(s))


def _rj(s, w):
    s = str(s)
    return " " * max(0, w - _dw(s)) + s


def _hb(b):
    """바이트를 사람이 읽는 고정폭 문자열로(GiB/MiB 자동, 우측정렬용)."""
    if b is None or b < 0:
        return "n/a"
    if b >= 0.5 * 1024 ** 3:
        return f"{b / 1024 ** 3:.3f} GiB"
    if b >= 1024 ** 2:
        return f"{b / 1024 ** 2:.1f} MiB"
    return f"{b / 1024:.1f} KiB"


# ──────────────────────────────────────────────────────────────────────────
# 이론 모델 — 빌드된 네트워크 트리에서 ①파라미터/②그래디언트/③옵티마이저를 '정확히',
#            ④액티베이션을 '추정'으로 계산해 4-버킷 이론값을 만든다.
# ──────────────────────────────────────────────────────────────────────────
def _walk_bytes(tree):
    """pytree / nnx.State 의 array leaf 들을 (원소수, 총바이트, {dtype명: 바이트}) 로 집계.
    dtype 별로 나눠 bf16(2B) / fp32(4B) 를 정확히 반영한다."""
    import jax
    import numpy as np
    if tree is None:
        return 0, 0, {}
    n = 0
    by = {}
    for l in jax.tree_util.tree_leaves(tree):
        if not (hasattr(l, "shape") and hasattr(l, "dtype")):
            continue
        sz = int(np.prod(l.shape))
        name = np.dtype(l.dtype).name
        by[name] = by.get(name, 0) + sz * int(np.dtype(l.dtype).itemsize)
        n += sz
    return n, int(sum(by.values())), by


def _resnet_v2_act_elems(H, W, stage_sizes, num_filters):
    """ResNetV2Encoder(networks/encoders.py) forward 가 backward 용으로 보존하는 feature-map
    원소 수(샘플 1개분, 근사). 토폴로지: stem(conv7x7 s2 → maxpool3x3 s2) → 각 stage 의 첫
    블록만 stride2 다운샘플, 블록당 conv 출력 2개 보존(GroupNorm/1x1 projection 은 무시한 하한)."""
    elems = 0
    h, w = H // 2, W // 2            # stem conv 7x7 stride 2 (224 → 112)
    elems += h * w * num_filters
    h, w = h // 2, w // 2            # maxpool 3x3 stride 2 (112 → 56)
    elems += h * w * num_filters
    for i, blocks in enumerate(stage_sizes):
        f = num_filters * (2 ** i)   # 64, 128, 256, 512
        if i > 0:                    # stage 1.. 첫 블록에서 stride 2
            h, w = h // 2, w // 2
        elems += blocks * 2 * h * w * f
    return int(elems)


def compute_theory(agent, args, batch_bytes, total_bs, minibatch_size):
    """빌드된 EXPO 에이전트의 실제 파라미터/옵티마이저 트리를 걸어 4-버킷 이론값 dict 생성.
    ①파라미터·②그래디언트·③옵티마이저는 트리에서 '정확'하고, ④액티베이션만 추정이다."""
    freeze = bool(args.freeze_actor)

    # ── ① 파라미터: 컴포넌트별 트리를 걸어 dtype별 바이트 ────────────────
    ats = agent.actor_train_state
    pi_params = ats.ema_params if getattr(ats, "ema_params", None) is not None else ats.params
    _, b_pi, by_pi = _walk_bytes(pi_params)
    b_pi_bf16 = by_pi.get("bfloat16", 0)
    b_pi_fp32 = b_pi - b_pi_bf16
    _, b_pi_opt, _ = _walk_bytes(getattr(ats, "opt_state", None))

    def ts_bytes(ts):
        _, bp, _ = _walk_bytes(ts.params)
        _, bo, _ = _walk_bytes(getattr(ts, "opt_state", None))
        return bp, bo

    b_crit_p, b_crit_o = ts_bytes(agent.critic)
    b_enc_p,  b_enc_o  = ts_bytes(agent.batch_encoder)
    b_res_p,  b_res_o  = ts_bytes(agent.residual_actor)
    b_tmp_p,  b_tmp_o  = ts_bytes(agent.temp)
    _, b_tcrit_p, _ = _walk_bytes(agent.target_critic.params)
    # target_actor_params: freeze/resume 면 pi0.5 와 버퍼 공유(추가 0), 아니면 별도 사본.
    _, b_tact_full, _ = _walk_bytes(agent.target_actor_params)
    b_tact_extra = 0 if freeze else b_tact_full

    enc_trainable = not bool(getattr(agent, "freeze_critic_encoder", False))
    # (라벨, 파라미터바이트, dtype, 학습대상?)  — 전부 '상주'.
    components = [
        ("pi0.5 base (frozen)",            b_pi_bf16,  "bf16", False),
        ("pi0.5 LoRA",                     b_pi_fp32,  "fp32", (not freeze)),
        ("critic ensemble (×%d)" % args.num_qs, b_crit_p, "fp32", True),
        ("critic encoder ResNetV2",        b_enc_p,    "fp32", enc_trainable),
        ("residual actor (TanhNormal)",    b_res_p,    "fp32", True),
        ("temperature (scalar)",           b_tmp_p,    "fp32", True),
        ("target_critic (사본)",           b_tcrit_p,  "fp32", False),
        ("target_actor (%s)" % ("pi0.5 공유→0" if freeze else "별도 사본"),
                                           b_tact_extra, "—" if freeze else "mix", False),
    ]
    params_total = sum(c[1] for c in components)

    # ── ② 그래디언트: 학습 대상 파라미터 크기만큼(backward 동안만 존재) ──
    grad_total = sum(c[1] for c in components if c[3])

    # ── ③ 옵티마이저(Adam m,v): 트리에서 실측. 참고로 Adam 이면 ≈ 2×학습대상 ──
    opt_total = b_crit_o + b_enc_o + b_res_o + b_tmp_o + b_pi_opt
    opt_formula = 2 * grad_total

    # ── ④ 액티베이션(추정) + 입력배치(정확) ─────────────────────────────
    # 입력 이미지배치는 arg 에 상주하며 '총배치(=batch_size×utd_ratio)'에 비례 → batch_bytes 로 정확.
    input_batch = int(batch_bytes)
    # 인코더/Q 액티베이션은 lax.scan 때문에 '미니배치(=batch_size)'에 비례.
    resnet_elems = _resnet_v2_act_elems(
        args.resize, args.resize,
        getattr(args, "encoder_stage_sizes", (3, 4, 6, 3)),
        getattr(args, "encoder_num_filters", 64),
    )
    enc_act = 2 * minibatch_size * resnet_elems * 4         # obs + next_obs 2회 인코딩 가정, fp32
    cand = args.N + args.n_edit_samples                     # critic 이 채점하는 후보 수
    hidden_sum = sum(getattr(args, "hidden_dims", (256, 256, 256)))
    q_act = minibatch_size * cand * args.num_qs * hidden_sum * 4
    activation_est = enc_act + q_act

    resident_theory = params_total + opt_total
    step_total_theory = resident_theory + input_batch + grad_total + activation_est

    return {
        "components": components,
        "params_total": params_total,
        "grad_total": grad_total,
        "opt_total": opt_total,
        "opt_formula": opt_formula,
        "pi_opt": b_pi_opt,
        "input_batch": input_batch,
        "enc_act": enc_act,
        "q_act": q_act,
        "activation_est": activation_est,
        "resident_theory": resident_theory,
        "step_total_theory": step_total_theory,
        "minibatch": int(minibatch_size),
        "total_bs": int(total_bs),
        "cand": int(cand),
        "resnet_elems": int(resnet_elems),
        "freeze": freeze,
    }


def _make_raw_observation(agent, resize):
    """rollout 경로(sample_actions)가 받는 raw env 관측 dict 합성 (uint8 카메라 + state)."""
    import numpy as np
    H = W = 256  # robocasa camera 256x256 (process_raw_inputs 가 내부에서 모델 해상도로 resize)
    return {
        "observation/image": np.random.randint(2, 256, (H, W, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(2, 256, (H, W, 3), dtype=np.uint8),
        "observation/right_image": np.random.randint(2, 256, (H, W, 3), dtype=np.uint8),
        "observation/state": np.zeros((16,), dtype=np.float32),
        "prompt": agent.default_prompt or "pick up the object",
    }


# ──────────────────────────────────────────────────────────────────────────
# WORKER — 한 변형을 실제로 빌드/실행하고 깨끗한 peak 을 측정해 result-file 에 기록
# ──────────────────────────────────────────────────────────────────────────
def run_worker(args):
    import gc
    import jax

    dev = jax.local_devices()[0]
    print("=" * 92)
    print(f"[worker phase={args.phase} freeze_actor={args.freeze_actor} n_edit={args.n_edit_samples}] "
          f"device={dev} platform={dev.platform}")
    lim = gpu_limit()
    if lim >= 0:
        print(f"GPU 메모리 예산(bytes_limit, MEM_FRACTION 반영): {gib(lim):.2f} GiB")
    print("=" * 92)

    result = {
        "phase": args.phase,
        "label": args.label,
        "freeze_actor": bool(args.freeze_actor),
        "n_edit_samples": args.n_edit_samples,
        "num_qs": args.num_qs,
        "N": args.N,
        "batch_size": args.batch_size,
        "utd_ratio": args.utd_ratio,
        "limit": lim,
        "ok": False,
    }

    try:
        # ── 1) 빌드 → 정적 상주 R 측정 ────────────────────────────────────
        agent, data_sharding, replicated_sharding = build_agent(args)
        jax.block_until_ready(jax.tree_util.tree_leaves(agent))
        gc.collect()
        resident = gpu_in_use()
        build_peak = gpu_peak()
        result["resident"] = resident
        result["build_peak"] = build_peak
        print(f"\n[정적] 빌드 후 상주 bytes_in_use R = {gib(resident):.2f} GiB "
              f"(빌드 중 peak {gib(build_peak):.2f} GiB)")

        if args.phase == "resident":
            # op 없이 상주만 보고.
            result["after"] = resident
            result["peak"] = build_peak
            result["ok"] = True

        elif args.phase == "update":
            # ── 2) 실제 학습 한 스텝 ──────────────────────────────────────
            total_bs = args.batch_size * args.utd_ratio
            batch = _make_openpi_batch(agent, total_bs, args.resize, data_sharding)
            actor_batch = _make_openpi_batch(agent, args.batch_size, args.resize, data_sharding)
            jax.block_until_ready(jax.tree_util.tree_leaves(batch))
            jax.block_until_ready(jax.tree_util.tree_leaves(actor_batch))
            batch_bytes = _tree_nbytes(batch) + _tree_nbytes(actor_batch)
            result["batch_bytes"] = batch_bytes

            # ── 이론(4-버킷): 빌드된 트리에서 ①파라미터/②그래디언트/③옵티마이저 정확 + ④추정 ──
            # (donation 회피용 target_actor_params=None 으로 끊기 '전'에 계산해야 ④/공유판정이 정확)
            minibatch_size = total_bs // args.utd_ratio   # = batch_size (scan 1회당 미니배치)
            try:
                result["theory"] = compute_theory(agent, args, batch_bytes, total_bs, minibatch_size)
            except Exception as e:
                import traceback
                result["theory_error"] = f"{type(e).__name__}: {e}"
                traceback.print_exc()

            gc.collect()
            before_update = gpu_in_use()

            # [donation 충돌 회피] freeze 모드에선 build_pi05 가 target_actor_params 를
            # actor_train_state 와 '같은 버퍼'로 참조 공유한다(~7GiB 절약). _update_jit 는
            # donate_argnums=(0,)로 self 를 통째로 donate 하는데, 같은 버퍼가 두 번 들어가면
            # "donate the same buffer twice" 로 죽는다. freeze 모드에선 target 이
            # update_actor(스킵됨)에서만 읽히는 죽은 참조라, 측정 직전에 끊어준다.
            # 버퍼 자체는 actor_train_state 에 그대로 남으므로 메모리(상주 R)는 변하지 않는다.
            if args.freeze_actor:
                agent = agent.replace(target_actor_params=None)

            # train_pi_robo.run_agent_updates 와 동일: rng 를 replicate 후 update 호출.
            agent = agent.replace(rng=jax.device_put(agent.rng, replicated_sharding))

            # ── (A) 컴파일된 update 의 '메모리 요구량' — 핵심 지표 ──────────
            # peak_bytes_in_use 는 빌드 시 fp32 모델 생성 transient(~15GiB)에 가려질 수 있다
            # (누적 최댓값이라 리셋 불가). 대신 컴파일된 실행파일의 memory_analysis 를 보면
            # 빌드와 무관하게 '이 update 한 번이 GPU 에서 동시에 잡아야 하는 버퍼'를 정확히 안다.
            #   arg  : 입력 전체(donate 되는 self=상주 파라미터 + batch). 이미 상주 중.
            #   temp : 실행 중 scratch(=backward 용 activation + gradient + XLA fusion). ★새로 드는 부분
            #   output: 출력(갱신된 train state + info). donation 으로 입력 버퍼 재사용분이 alias.
            #   총요구 = arg + temp + output − alias  (XLA 가 실제로 동시에 잡는 live peak)
            try:
                import openpi.shared.array_typing as at
                ag = agent.replace(_infer_cache=None)
                # openpi TrainState 는 런타임 타입체크(jaxtyping/beartype)가 걸려 있어, AOT
                # .lower() 가 추상 ArgInfo 로 TrainState 를 만들 때 타입체크가 터진다. 실제
                # jit 실행 경로엔 영향 없으니 lower/compile 동안만 끈다(_split_params 와 동일 패턴).
                with at.disable_typechecking():
                    compiled = type(agent)._update_jit.lower(
                        ag, batch, args.utd_ratio, actor_batch
                    ).compile()
                ms = compiled.memory_analysis()
                mem = {
                    "arg": int(getattr(ms, "argument_size_in_bytes", 0) or 0),
                    "temp": int(getattr(ms, "temp_size_in_bytes", 0) or 0),
                    "output": int(getattr(ms, "output_size_in_bytes", 0) or 0),
                    "alias": int(getattr(ms, "alias_size_in_bytes", 0) or 0),
                    "gencode": int(getattr(ms, "generated_code_size_in_bytes", 0) or 0),
                }
                mem["total"] = mem["arg"] + mem["temp"] + mem["output"] - mem["alias"]
                result["mem"] = mem
                print(f"[update/compiled] arg={gib(mem['arg']):.2f}  temp(activation+grad+scratch)="
                      f"{gib(mem['temp']):.2f}  output={gib(mem['output']):.2f}  alias={gib(mem['alias']):.2f}")
                print(f"[update/compiled] → 한 스텝 총 메모리 요구 = {gib(mem['total']):.2f} GiB "
                      f"(상주 R 대비 +{gib(mem['total'] - resident):.2f} GiB)")
            except Exception as e:
                result["mem_error"] = f"{type(e).__name__}: {e}"
                print(f"[update/compiled] memory_analysis 실패(런타임 peak 로 대체): {result['mem_error']}")

            # ── (B) 실제 실행 — 런타임 peak/after (참고; 빌드 transient 에 가려질 수 있음) ──
            # compiled 총요구가 예산을 넘으면 실제 실행은 '확정 OOM'이라 굳이 돌려 프로세스를
            # 죽이지 않는다(이론 + compiled 총요구만으로 비교는 충분). --force-run 으로 강제 가능.
            projected = result.get("mem", {}).get("total", -1)
            would_oom = (projected > 0 and lim > 0 and projected > 0.95 * lim)
            if would_oom and not args.force_run:
                msg = (f"compiled 총요구 {gib(projected):.2f}G > 0.95×예산 {gib(lim):.2f}G "
                       f"→ 실제 실행 생략(확정 OOM 회피, --force-run 으로 강제). 이론/compiled 로 비교.")
                result.update(after=resident, peak=build_peak, before_update=before_update,
                              ok=True, run_skipped=msg)
                print(f"[update] {msg}")
            else:
                print(f"[update] agent.update 실행 (총 배치 {total_bs} = bs {args.batch_size} × utd {args.utd_ratio}) ...")
                new_agent, info = agent.update(agent, batch, args.utd_ratio, actor_batch)
                jax.block_until_ready(jax.tree_util.tree_leaves(new_agent))
                jax.block_until_ready([v for v in info.values() if hasattr(v, "shape")])

                after = gpu_in_use()
                peak = gpu_peak()
                result.update(after=after, peak=peak, before_update=before_update, ok=True)
                print(f"[update] 런타임: 직전 {gib(before_update):.2f} GiB → 직후 {gib(after):.2f} GiB, "
                      f"누적 peak {gib(peak):.2f} GiB (빌드 transient 포함)")

        elif args.phase == "sample":
            # ── 2') rollout 추론 한 번 (후보 N개 propose→edit→select) ────
            norm_stats = getattr(agent.actor.data_config, "norm_stats", None)
            if norm_stats is None:
                result["skipped"] = "norm_stats 없음(--random-init). rollout 측정은 --ckpt 필요."
                result["after"] = resident
                result["peak"] = build_peak
                result["ok"] = True
                print(f"[sample] 건너뜀: {result['skipped']}")
            else:
                obs = _make_raw_observation(agent, args.resize)
                gc.collect()
                before_sample = gpu_in_use()
                print("[sample] agent.sample_actions 호출 (첫 호출은 컴파일 포함) ...")
                action, agent, sinfo = agent.sample_actions(obs)
                jax.block_until_ready(action)
                after = gpu_in_use()
                peak = gpu_peak()
                result.update(after=after, peak=peak, before_update=before_sample, ok=True)
                print(f"[sample] 완료: peak {gib(peak):.2f} GiB, spike(peak − R) {gib(peak - resident):.2f} GiB")
        else:
            raise ValueError(f"unknown phase: {args.phase}")

    except Exception as e:
        import traceback
        result["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()

    # 결과를 result-file(JSON)로 — 부모(드라이버)가 읽는다.
    if args.result_file:
        with open(args.result_file, "w") as f:
            json.dump(result, f)
    print("RESULT_JSON " + json.dumps(result))
    return 0 if result.get("ok") else 1


# ──────────────────────────────────────────────────────────────────────────
# DRIVER — 변형들을 독립 서브프로세스로 순차 실행하고 표/해설 출력
# ──────────────────────────────────────────────────────────────────────────
def _worker_argv(args, phase, label, freeze_actor, n_edit_samples, result_file):
    """현재 빌드 플래그를 그대로 물려 worker 서브프로세스용 argv 를 만든다."""
    argv = [
        sys.executable, os.path.abspath(__file__), "--worker",
        "--phase", phase, "--label", label,
        "--freeze-actor", str(int(freeze_actor)),
        "--n-edit-samples", str(n_edit_samples),
        "--N", str(args.N),
        "--num-qs", str(args.num_qs),
        "--edit-scale", str(args.edit_scale),
        "--batch-size", str(args.batch_size),
        "--utd-ratio", str(args.utd_ratio),
        "--replan-steps", str(args.replan_steps),
        "--resize", str(args.resize),
        "--pi05-config-name", args.pi05_config_name,
        "--fsdp-devices", str(args.fsdp_devices),
        "--seed", str(args.seed),
        "--result-file", result_file,
    ]
    if args.random_init:
        argv.append("--random-init")
    else:
        argv += ["--ckpt", args.ckpt]
    if getattr(args, "force_run", False):
        argv.append("--force-run")
    return argv


def _run_one(args, phase, label, freeze_actor, n_edit_samples):
    """worker 하나를 서브프로세스로 실행하고 결과 dict 를 돌려준다(자식 로그는 그대로 흘려보냄)."""
    fd, path = tempfile.mkstemp(suffix=".json", prefix="vramtrain_")
    os.close(fd)
    try:
        argv = _worker_argv(args, phase, label, freeze_actor, n_edit_samples, path)
        print("\n" + "#" * 92)
        print(f"# 서브프로세스 실행: [{label}]  (phase={phase}, freeze_actor={freeze_actor}, n_edit={n_edit_samples})")
        print("#" * 92, flush=True)
        proc = subprocess.run(argv, env=os.environ.copy())
        try:
            with open(path) as f:
                res = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            res = {"label": label, "phase": phase, "ok": False,
                   "error": f"worker 비정상 종료(returncode={proc.returncode})"}
        res["returncode"] = proc.returncode
        return res
    finally:
        if os.path.exists(path):
            os.remove(path)


def _signed_gib(b):
    """+/− 부호 + GiB(2자리). 이론-실측 차이 표기용."""
    sign = "+" if b >= 0 else "−"
    return f"{sign}{abs(b) / 1024 ** 3:.2f}G"


# ──────────────────────────────────────────────────────────────────────────
# 출력 블록 — 한글 전각폭(_lj/_rj)으로 열을 실제로 맞춘다.
# ──────────────────────────────────────────────────────────────────────────
def print_theory_block(th, lim):
    """[이론] 4-버킷 표. ①②③ 은 트리에서 정확, ④ 는 추정."""
    if not th:
        print("\n[이론] 계산 실패(theory 없음).")
        return
    L, B, D, X = 38, 13, 7, 18
    W = L + B + D + X

    def row(label, b=None, dt="", dep=""):
        print(_lj(label, L) + _rj(_hb(b) if b is not None else "", B)
              + _rj(dt, D) + _rj(dep, X))

    print("\n" + "═" * W)
    print(_lj(" [이론] 학습 1스텝 VRAM = ①파라미터+②그래디언트+③옵티마이저+④액티베이션", W))
    print("═" * W)
    print(_lj("  버킷 / 구성", L) + _rj("바이트", B) + _rj("dtype", D) + _rj("batch依存", X))
    print("─" * W)
    # ① 파라미터
    print(_lj(" ① 파라미터(weights) — 상주", W))
    for label, b, dt, _trainable in th["components"]:
        row("     " + label, b, dt, "no")
    row("     ─ 소계 (resident)", th["params_total"], "", "no")
    # ② 그래디언트
    print(_lj(" ② 그래디언트 — backward 동안만(=학습대상 크기)", W))
    row("     critic+encoder+residual+temp", th["grad_total"], "fp32", "no")
    if th["freeze"]:
        row("     pi0.5 (freeze → grad 0)", 0, "—", "no")
    # ③ 옵티마이저
    print(_lj(" ③ 옵티마이저 Adam(m,v) — 상주", W))
    row("     학습대상 2벌 (트리 실측)", th["opt_total"], "fp32", "no")
    row("     (참고: 2×② 공식값)", th["opt_formula"], "fp32", "no")
    # ④ 액티베이션 + 입력배치
    print(_lj(" ④ 액티베이션 + 입력배치 — ★batch 비례(transient)", W))
    row("     · 입력 이미지배치(arg, 정확)", th["input_batch"], "fp32", f"∝총배치 {th['total_bs']}")
    row("     · 인코더 act(temp, 추정)", th["enc_act"], "fp32", f"∝미니배치 {th['minibatch']}")
    row("     · Q앙상블 act(temp, 추정)", th["q_act"], "fp32", "∝미니배치")
    row(f"     · pi0.5 forward scratch(미모델)", None, "", "실측 temp 에 포함")
    row("     ─ ④ 소계(모델링분만)", th["input_batch"] + th["activation_est"], "", "∝batch")
    print("─" * W)
    row(" 상주 R(이론) = ①+③", th["resident_theory"], "", "고정")
    row(" 1스텝 총요구(이론 하한) ①+②+③+④", th["step_total_theory"], "", "④ 일부 미모델")
    if lim and lim > 0:
        # 하한값이라: 하한>예산이면 '확정 OOM', 하한<예산이면 '미모델 ④ 로 초과 가능'.
        verdict = "하한>예산→확정OOM" if th["step_total_theory"] > lim else "하한<예산(미모델④로 초과가능)"
        row(" GPU 예산(bytes_limit)", lim, "", verdict)
    print("═" * W)
    print(" ※ ①②③ 은 트리에서 '정확'. ④ 는 추정이며 pi0.5 frozen forward(후보 N×미니배치를 3B 모델에")
    print("    통과)·attention seq² scratch 가 미모델이라 '하한'이다 → 실제 ④ 는 실측 temp 로만 정확.")
    print(" ※ ①③ 은 batch 무관(고정), ④ 만 batch 와 함께 출렁 → batch 를 줄이면 OOM 이 풀리는 이유.")
    print(" ※ utd_ratio 가 배치를 미니배치로 scan → 보존 act 는 미니배치(=batch_size), 입력배치만 총배치(=bs×utd).")


def print_actual_block(r, lim):
    """[실측] compiled memory_analysis + bytes_in_use."""
    L, V, N = 42, 14, 20
    W = L + V + N

    def row(label, b=None, note=""):
        print(_lj(label, L) + _rj(_hb(b) if b is not None else "", V) + _rj(note, N))

    mem = r.get("mem") or {}
    print("\n" + "═" * W)
    print(_lj(" [실측] compiled memory_analysis + bytes_in_use", W))
    print("═" * W)
    row("  상주 R (bytes_in_use, 빌드후)", r.get("resident"), "→ 이론 ①+③")
    if mem:
        row("  compiled arg (donate self+입력배치)", mem.get("arg"))
        row("  compiled temp (act+grad+scratch)", mem.get("temp"), "→ 이론 ②+④")
        row("  compiled output (갱신 train state)", mem.get("output"))
        row("  compiled alias (donation 재사용)", mem.get("alias"))
        row("  compiled 총요구 = arg+temp+out−alias", mem.get("total"), "★권위 지표")
    else:
        print("  (compiled memory_analysis 없음: " + str(r.get("mem_error", "")) + ")")
    if r.get("run_skipped"):
        print(_lj("  runtime: 생략", L) + _rj("", V) + _rj("", N))
        print("    └ " + r["run_skipped"])
    else:
        row("  runtime peak (빌드 transient 포함, 참고)", r.get("peak"))
    if lim and lim > 0:
        row("  GPU 예산 (bytes_limit)", lim)
    print("═" * W)


def print_bridge_block(r):
    """[비교] 이론 vs 실측. 핵심 다리: 실측 액티베이션 = compiled temp − 이론 ②그래디언트."""
    th = r.get("theory")
    mem = r.get("mem") or {}
    if not th:
        return
    L, T, A, NT = 28, 14, 14, 26
    W = L + T + A + NT

    def row(item, t=None, a=None, note=""):
        print(_lj(item, L) + _rj(_hb(t) if t is not None else "", T)
              + _rj(_hb(a) if a is not None else "", A) + _rj(note, NT))

    print("\n" + "═" * W)
    print(_lj(" [비교] 이론 vs 실측 (bridge)", W))
    print("═" * W)
    print(_lj("  항목", L) + _rj("이론", T) + _rj("실측", A) + _rj("주석", NT))
    print("─" * W)
    R = r.get("resident", -1)
    row("  상주 R = ①+③", th["resident_theory"], R if R >= 0 else None,
        (_signed_gib(R - th["resident_theory"]) + " (CUDA ctx 등)") if R >= 0 else "")
    if mem and R >= 0:
        row("  입력 이미지배치", th["input_batch"], mem["arg"] - R, "실측 = arg − 상주R")
        row("  액티베이션 ④", th["activation_est"], mem["temp"] - th["grad_total"],
            "실측 = temp − ②grad")
        row("  그래디언트 ②", th["grad_total"], None, "(실측 temp 안에 포함)")
        row("  1스텝 총요구", th["step_total_theory"], mem["total"],
            _signed_gib(mem["total"] - th["step_total_theory"]))
    print("═" * W)
    if mem:
        print("  · '입력배치/액티베이션' 실측은 compiled arg·temp 에서 이론 ①③/②를 빼 역산한 값.")
        print("  · 이론 ④(추정)와 실측 액티베이션이 다르면 XLA fusion·remat·후보(N+n_edit) 중간텐서·")
        print("    pi0.5 frozen forward scratch 차이 때문(추정은 하한, 실측이 권위).")


def print_interpretation(th, lim):
    """4-버킷 관점의 해석 + nvidia-smi 주의."""
    print("\n해석 — 학습 VRAM 을 4개로 나눠 보면:")
    print("  ① 파라미터(상주)   : pi0.5 base(bf16)가 절대다수. frozen 이라도 '가중치'는 GPU 에 상주한다.")
    print("  ② 그래디언트       : 학습대상(critic/encoder/residual/temp, fp32)만큼. backward 동안만.")
    print("                       freeze_pi05_actor=True → pi0.5 는 grad 0(가장 큰 절감 포인트).")
    print("  ③ 옵티마이저(상주) : Adam m,v 2벌(학습대상 fp32). pi0.5 는 set_to_zero 라 ~0.")
    print("  ④ 액티베이션       : ★batch·해상도에 비례(transient). update peak 의 변동분 대부분.")
    print("                       critic ResNetV2×3cam + Q앙상블(num_qs) + 후보(N+n_edit) forward/backward.")
    print("  · ①③ 은 고정, ④ 만 출렁 → OOM 이면 batch_size·N·n_edit·num_qs·utd_ratio 를 낮추는 게 ④ 직접 절감.")
    if lim and lim > 0 and th and th["step_total_theory"] > lim:
        print(f"  · ★ 이번 설정의 1스텝 총요구(이론 {gib(th['step_total_theory']):.1f}G)가 예산({gib(lim):.1f}G)을")
        print(f"      넘는다 → run_server.sh 가 freeze·num_qs·N·n_edit 를 줄여 ④ 를 예산 아래로 누르는 이유.")
    print("\n주의 — nvidia-smi 의 '~30GB/32GB' 와 헷갈리지 말 것:")
    print("  · XLA_PYTHON_CLIENT_MEM_FRACTION(0.9)은 시작 시 GPU 90% 를 '미리' 선점 → nvidia-smi 의 큰 숫자는")
    print("    '예약'이지 실제 사용량이 아니다. OOM 판정은 'compiled 총요구 vs bytes_limit'으로 한다.")


def run_driver(args):
    # 어떤 변형들을 돌릴지 결정.
    #   기본       : REAL(런과 동일) update 한 번.
    #   --breakdown: critic-only / REAL / unfreeze + (실 체크포인트면) rollout sample.
    runs = []
    if args.breakdown:
        runs.append(("update", "critic-only (freeze actor, n_edit=0)", True, 0))
        runs.append(("update", f"REAL (freeze actor, n_edit={args.n_edit_samples}) = run_server.sh", True, args.n_edit_samples))
        runs.append(("update", f"unfreeze actor (n_edit={args.n_edit_samples})", False, args.n_edit_samples))
        if not args.random_init:
            runs.append(("sample", "rollout sample_actions (propose→edit→select)", True, args.n_edit_samples))
    else:
        freeze = not args.unfreeze_actor
        label = ("REAL (freeze actor)" if freeze else "unfreeze actor (pi0.5 base 학습)")
        runs.append(("update", f"{label}, n_edit={args.n_edit_samples}", freeze, args.n_edit_samples))

    print("=" * 92)
    print("test_vram_train.py — 학습 시 VRAM 증가 측정 (드라이버)")
    print(f"  가중치 소스 : {'랜덤 init(체크포인트 불필요)' if args.random_init else args.ckpt}")
    print(f"  공통 설정   : num_qs={args.num_qs}, N={args.N}, batch_size={args.batch_size}, "
          f"utd_ratio={args.utd_ratio}, replan_steps={args.replan_steps}, resize={args.resize}, "
          f"pi05={args.pi05_config_name}")
    print(f"  변형 {len(runs)}개를 서브프로세스로 순차 실행(각자 깨끗한 peak 확보).")
    print("=" * 92)

    results = [_run_one(args, *r) for r in runs]
    lim = next((r.get("limit", -1) for r in results if r.get("limit", -1) >= 0), -1)

    # ── 단일 런(기본): 이론 → 실측 → 비교 → 해석 (run_server.sh 파라미터 그대로) ──
    # 이론은 빌드만 성공하면 나오므로, 실제 update 가 OOM 으로 죽어도(ok=False) 이론/compiled 는 보여준다.
    if not args.breakdown:
        r0 = results[0] if results else {}
        if r0.get("theory"):
            if not r0.get("ok"):
                print(f"\n[주의] worker 가 끝까지 못 감(error={r0.get('error') or r0.get('mem_error')}). "
                      f"아래는 빌드 직후까지 확보한 이론/실측이다.")
            print_theory_block(r0["theory"], lim)
            print_actual_block(r0, lim)
            print_bridge_block(r0)
            print_interpretation(r0.get("theory"), lim)
            print("═" * 84)
            return 0
        # 이론조차 없으면(빌드 실패 등) 아래 기존 요약표로 폴백.

    # ── 최종 표 (--breakdown 또는 폴백) ──────────────────────────────────
    # 핵심 지표는 '컴파일된 update 총 요구(=arg+temp+output−alias)'. temp 가 학습이 새로
    # 추가하는 부분(activation+grad+scratch). 런타임 peak 은 빌드 transient 에 가려질 수 있어 참고용.
    def total_req(r):
        m = r.get("mem")
        return m["total"] if m else -1

    def temp_req(r):
        m = r.get("mem")
        return m["temp"] if m else -1

    print("\n" + "=" * 100)
    print("요약: 학습 한 스텝이 GPU 에서 동시에 잡아야 하는 VRAM")
    print("=" * 100)
    print(f"{'변형':<40}{'상주 R':>11}{'compiled 총요구':>17}{'그중 temp':>13}{'총−R':>11}{'런타임peak':>13}")
    print("-" * 100)
    for r in results:
        if not r.get("ok"):
            print(f"{r.get('label','?'):<40}  실패: {r.get('error','')}")
            continue
        R = r.get("resident", -1)
        T = total_req(r)
        tmp = temp_req(r)
        P = r.get("peak", -1)
        skip = r.get("skipped")
        if skip:
            print(f"{r['label']:<40}{gib(R):>8.2f}G   (skip: {skip})")
        elif T >= 0:
            print(f"{r['label']:<40}{gib(R):>8.2f}G{gib(T):>14.2f}G{gib(tmp):>11.2f}G"
                  f"{gib(T - R):>9.2f}G{gib(P):>11.2f}G")
        else:
            print(f"{r['label']:<40}{gib(R):>8.2f}G{'  (compiled n/a)':>14}{'':>11}{'':>9}{gib(P):>11.2f}G")
    print("-" * 100)
    lim = next((r.get("limit", -1) for r in results if r.get("limit", -1) >= 0), -1)
    if lim >= 0:
        print(f"GPU 메모리 예산(bytes_limit) = {gib(lim):.2f} GiB  "
              f"(compiled 총요구가 이걸 넘으면 OOM / XLA remat-abort)")
    print("  · compiled 총요구 = arg(상주 파라미터+batch) + temp(activation+grad+scratch) + output − alias")
    print("  · temp = '학습이 정적 모델 위에 새로 얹는' 메모리의 핵심(=backward activation+gradient).")

    # ── 컴포넌트 분해 해설 (--breakdown) ─────────────────────────────────
    by_label = {r.get("label"): r for r in results if r.get("ok")}

    def _tot(lbl):
        r = by_label.get(lbl)
        return total_req(r) if r else -1

    if args.breakdown:
        crit = _tot("critic-only (freeze actor, n_edit=0)")
        real = _tot(f"REAL (freeze actor, n_edit={args.n_edit_samples}) = run_server.sh")
        unfz = _tot(f"unfreeze actor (n_edit={args.n_edit_samples})")
        R_real = next((r.get("resident", -1) for r in results
                       if r.get("label", "").startswith("REAL")), -1)
        R_unfz = next((r.get("resident", -1) for r in results
                       if r.get("label", "").startswith("unfreeze")), -1)
        def _signed(b):
            """+/− 부호 + GiB. |delta| 가 0.2GiB 미만이면 'XLA noise 수준'으로 표기."""
            s = f"{'+' if b >= 0 else '−'}{gib(abs(b)):.2f} GiB"
            return s + ("  (≈0, XLA buffer-assign 노이즈 수준)" if abs(b) < 0.2 * 1024 ** 3 else "")

        print("\n컴포넌트별 기여(컴파일 총요구 차이로 분해):")
        if crit >= 0 and R_real >= 0:
            print(f"  · critic + 후보샘플링(pi0.5 forward, ResNet×3cam×Q앙상블) : "
                  f"총요구 {gib(crit):.2f} GiB  (상주 R 대비 {_signed(crit - R_real)})")
        if crit >= 0 and real >= 0:
            print(f"  · residual actor + temperature 추가분                     : "
                  f"{_signed(real - crit)}  → REAL(run_server.sh) 총요구 {gib(real):.2f} GiB")
        if real >= 0 and unfz >= 0:
            d_res = (R_unfz - R_real) if (R_unfz >= 0 and R_real >= 0) else -1
            print(f"  · pi0.5 action expert backward(update_actor) 추가분       : "
                  f"{_signed(unfz - real)} (activation)")
            if d_res >= 0:
                print(f"      └ 게다가 상주 R 도 +{gib(d_res):.2f} GiB (Adam m,v + target 사본). "
                      f"freeze_pi05_actor 가 이 둘을 모두 제거.")
        elif unfz < 0:
            print("  · pi0.5 action expert backward(unfreeze): 이 GPU 에선 빌드 단계에서 OOM "
                  "(=full 파인튜닝이 안 들어감 → run_server.sh 가 freeze 를 쓰는 이유). "
                  "여유 있는 GPU 에서 재측정 권장.")
        print("  ※ batch_size 가 작으면(예: 2) 컴포넌트 delta 가 XLA 버퍼배치 노이즈에 묻힌다.")
        print("    의미 있는 분해는 실제 런 크기로 --batch-size 를 키워서 볼 것.")

        # REAL(run_server.sh) 변형의 4-버킷 이론/실측 비교도 함께 보여준다.
        real_r = next((r for r in results if r.get("label", "").startswith("REAL")), None)
        if real_r and real_r.get("theory"):
            print_theory_block(real_r["theory"], lim)
            print_bridge_block(real_r)

    # ── 해설 ──────────────────────────────────────────────────────────────
    print("\n해석 — 왜 학습할 때 VRAM 이 상주 R 위로 더 올라가나:")
    print("  1) forward activation: backward 에서 쓸 중간값을 전부 들고 있어야 함(batch_size·")
    print("     이미지 해상도·모델 깊이에 비례). update peak 의 대부분이 여기서 나온다.")
    print("  2) gradient 1벌: 학습가능 파라미터 크기만큼 추가(적용 후 해제).")
    print("  3) 후보 샘플링: critic 미니배치마다 pi0.5 가 (N+n_edit)개 forward → activation.")
    print("     run_server.sh 가 N·n_edit·num_qs 를 줄인 이유가 바로 이 update peak 억제.")
    print("  4) critic Q앙상블 × ResNetV2 × 카메라 3대(9채널) forward+backward.")
    print("  5) freeze_pi05_actor=False 면 pi0.5 action expert backward 가 더해져 가장 크게 뜀")
    print("     (+옵티마이저 m,v 와 target 사본까지 상주 R 도 증가).")
    print("  · 이 activation 들은 jit 함수가 끝나면 해제돼 bytes_in_use 는 R 로 돌아온다. 하지만")
    print("    그 사이의 peak 가 GPU 예산을 넘으면 OOM / XLA remat-abort 가 난다(정적 R 이")
    print("    여유 있어 보여도 update 에서 터지는 이유). 줄이려면 batch_size·N·n_edit·num_qs·")
    print("    utd_ratio 를 낮추거나 freeze_pi05_actor 로 (5)를 제거한다.")
    print("\n주의 — nvidia-smi 의 '~30GB/32GB' 와 헷갈리지 말 것:")
    print("  · XLA_PYTHON_CLIENT_MEM_FRACTION(예: 0.9)은 시작 시 GPU 의 90% 를 '미리 통째로'")
    print("    선점한다. nvidia-smi 가 보여주는 큰 숫자는 이 '예약'이지 실제 사용량이 아니다.")
    print("  · 이 스크립트의 상주 R / compiled 총요구 는 bytes_in_use 기반의 '실제 live 사용량'")
    print("    이다. OOM 여부는 'compiled 총요구 vs bytes_limit(=예산)'으로 판단한다.")
    print("  · 표의 '런타임peak' 는 빌드 시 fp32 모델 생성 transient(~15GiB)가 섞일 수 있어")
    print("    참고용. 권위 있는 지표는 'compiled 총요구'다.")
    print("=" * 92)
    return 0


# ──────────────────────────────────────────────────────────────────────────
def build_arg_parser():
    ap = argparse.ArgumentParser(description="실제 학습(update/sample) 시 VRAM 증가 측정")
    # 동작 모드
    ap.add_argument("--worker", action="store_true", help="(내부용) 단일 변형을 직접 측정")
    ap.add_argument("--phase", default="update", choices=["update", "sample", "resident"],
                    help="worker 가 측정할 작업")
    ap.add_argument("--label", default="", help="(내부용) 표에 쓸 라벨")
    ap.add_argument("--result-file", default="", help="(내부용) 결과 JSON 기록 경로")
    ap.add_argument("--breakdown", action="store_true",
                    help="critic-only/REAL/unfreeze(+rollout) 분해 측정")
    ap.add_argument("--unfreeze-actor", action="store_true",
                    help="단일 측정 시 pi0.5 base actor 까지 학습(무거운 경우)")
    ap.add_argument("--force-run", action="store_true",
                    help="compiled 총요구가 예산을 넘어도 실제 agent.update 를 강제 실행(기본은 OOM 회피로 생략)")

    # 가중치 소스
    ap.add_argument("--random-init", action="store_true",
                    help="체크포인트 없이 랜덤 init(메모리는 실제와 동일). 기본은 --ckpt 사용")
    ap.add_argument("--ckpt", default="./checkpoints/robocasa365/pi05_pretrain_human300/multitask_learning/75000",
                    help="실제 체크포인트 루트(params/, assets/ 포함). run_server.sh 의 $CKPT")

    # 빌드 하이퍼파라미터 (기본 = scripts/sim/run_server.sh 의 robocasa 실제 런)
    ap.add_argument("--num-qs", type=int, default=10)
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--n-edit-samples", type=int, default=8)
    ap.add_argument("--edit-scale", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--utd-ratio", type=int, default=20)
    ap.add_argument("--replan-steps", type=int, default=8)
    ap.add_argument("--resize", type=int, default=224, help="critic 입력 이미지 해상도(모델 224)")
    ap.add_argument("--pi05-config-name", default="pi05_pretrain_human300_lora")
    ap.add_argument("--fsdp-devices", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    # worker 가 직접 받는 freeze 플래그(드라이버가 변형별로 채움). 0/1.
    ap.add_argument("--freeze-actor", type=int, default=1)
    return ap


def main():
    args = build_arg_parser().parse_args()
    if args.worker:
        # worker: --freeze-actor 로 받은 값을 freeze_actor 로 사용.
        args.freeze_actor = int(args.freeze_actor)
        return run_worker(args)
    # driver: 단일 모드의 freeze 는 --unfreeze-actor 로 결정.
    args.freeze_actor = 0 if args.unfreeze_actor else 1
    return run_driver(args)


if __name__ == "__main__":
    sys.exit(main())
