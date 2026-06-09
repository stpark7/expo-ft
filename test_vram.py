#!/usr/bin/env python
"""test_vram.py — EXPO-FT의 각 신경망이 GPU VRAM을 얼마나 먹는지 컴포넌트별로 수치 측정.

무엇을 증명하나
---------------
"pi0.5는 frozen이니 VRAM을 적게 쓴다"는 직관이 맞는지 컴포넌트별로 실측한다. 두 수치를
함께 보고한다.

  1) 파라미터 바이트(이론값): pytree leaf의 shape×dtype 합. 체크포인트/데이터 불필요, 결정적.
  2) 실제 GPU 점유 증가량 Δ: 컴포넌트를 GPU에 올린 직후 jax memory_stats()의
     bytes_in_use 증가분. XLA가 실제로 잡은 live buffer라 '진짜로 먹는 VRAM'이다.

pi0.5는 사전학습 체크포인트 없이 '구조만' 랜덤 초기화로 만든다(shape/dtype는 학습과 동일,
값만 랜덤이라 메모리에는 영향 없음). frozen base는 학습 경로(pi05_init_train_state)와 똑같이
bf16으로 캐스팅하고, 학습 가능한 LoRA 어댑터만 fp32로 남긴다.

EXPO 쪽 신경망(critic encoder / critic ensemble / residual actor / temperature)은
EXPOLearner.create()의 빌드 코드를 그대로 복제해 만든다.

실행
----
  source .venv/bin/activate
  CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python test_vram.py

옵션(기본값 = robocasa PickPlaceCounterToStove + expo_ft_pi_config 기준):
  --num-qs N        Q-앙상블 개수 (기본: config의 10. run_server.sh는 3)
  --replan-steps N  full_action_dim = replan_steps * action_dim (기본 8)
  --action-dim N    RL action 차원(arm) (기본 7)
  --state-dim N     critic state 차원 (기본 16; 메모리에 거의 영향 없음)
  --resize N        이미지 해상도 (기본 224)
  --num-cameras N   critic가 채널방향으로 쌓는 카메라 수 (기본 3 → 9채널)
  --no-pi05         pi0.5는 건너뛰고 EXPO 쪽 작은 신경망만 측정
"""

import argparse
import gc

import numpy as np
import jax
import jax.numpy as jnp


# ──────────────────────────────────────────────────────────────────────────
# 측정 유틸
# ──────────────────────────────────────────────────────────────────────────
def gpu_in_use() -> int:
    """현재 GPU가 실제로 잡고 있는 live buffer 바이트(bytes_in_use). CPU면 -1."""
    try:
        stats = jax.local_devices()[0].memory_stats()
    except Exception:
        return -1
    if not stats:
        return -1
    return int(stats.get("bytes_in_use", -1))


def gpu_peak() -> int:
    try:
        stats = jax.local_devices()[0].memory_stats() or {}
    except Exception:
        return -1
    return int(stats.get("peak_bytes_in_use", -1))


def gpu_limit() -> int:
    try:
        stats = jax.local_devices()[0].memory_stats() or {}
    except Exception:
        return -1
    return int(stats.get("bytes_limit", -1))


def leaves(tree):
    """pytree(또는 nnx State)에서 array leaf만 추출. nnx VariableState는 array를 leaf로 노출."""
    return [
        l for l in jax.tree_util.tree_leaves(tree)
        if hasattr(l, "shape") and hasattr(l, "dtype")
    ]


def tree_stats(tree):
    """(파라미터 개수, 바이트). dtype별 바이트를 정확히 반영(bf16=2, fp32=4)."""
    ls = leaves(tree)
    n = sum(int(np.prod(l.shape)) for l in ls)
    b = sum(int(np.prod(l.shape)) * np.dtype(l.dtype).itemsize for l in ls)
    return n, b


def mib(b):
    return b / (1024 ** 2)


def gib(b):
    return b / (1024 ** 3)


def realize(tree):
    """모든 leaf를 GPU에 확정 적재(지연 할당 방지)."""
    jax.block_until_ready(leaves(tree))
    return tree


KEEP = []   # GC로 버퍼가 해제되지 않도록 모든 파라미터를 붙들어 둔다.
ROWS = []   # 최종 표에 들어갈 (이름, 파라미터수, 바이트, GPU Δ)


def measure(name: str, build_fn):
    """build_fn()으로 파라미터를 만들고, 이론 바이트 + 실제 GPU Δ를 기록."""
    gc.collect()
    before = gpu_in_use()
    params = realize(build_fn())
    after = gpu_in_use()
    KEEP.append(params)
    n, b = tree_stats(params)
    delta = (after - before) if (before >= 0 and after >= 0) else -1
    ROWS.append((name, n, b, delta))
    d_str = f"{mib(delta):8.1f} MiB" if delta >= 0 else "   (CPU)"
    print(f"  [OK] {name:<34} params={n:>13,}  이론={mib(b):8.1f} MiB  GPU Δ={d_str}")
    return params


# ──────────────────────────────────────────────────────────────────────────
# pi0.5 (VLA base policy) — 구조만 랜덤 초기화, frozen base는 bf16 캐스팅
# ──────────────────────────────────────────────────────────────────────────
def measure_pi05():
    import flax.nnx as nnx
    import openpi.training.config as _config
    import openpi.shared.nnx_utils as nnx_utils

    cfg_name = "pi05_pretrain_human300_lora"
    print(f"\n[pi0.5] openpi config '{cfg_name}' 로드 → 모델 구조 생성(랜덤 init, 체크포인트 불필요)")
    train_config = _config.get_config(cfg_name)

    before = gpu_in_use()

    # pi05_init_train_state.init()과 동일: 모델 생성 후 frozen 파라미터를 bf16으로 캐스팅.
    # 주의: model.create()는 일단 fp32로 만들고(약 12.7 GiB) 그 뒤 frozen을 bf16으로 캐스팅한다.
    # 원본 fp32 model을 붙들고 있으면 fp32+bf16이 동시에 잡혀 Δ가 과대계상되므로, 캐스팅 후
    # model을 즉시 해제(del+gc)해 fp32 원본 버퍼를 비운다 → 학습 시 실제 상주(bf16 base+fp32 LoRA)와 일치.
    rng = jax.random.PRNGKey(0)
    model = train_config.model.create(rng)
    params = nnx_utils.state_map(
        nnx.state(model),
        train_config.freeze_filter,
        lambda p: p.replace(p.value.astype(jnp.bfloat16)),
    )
    realize(params)
    del model
    gc.collect()
    after = gpu_in_use()
    KEEP.append(params)

    # 학습 가능(LoRA 등) vs 동결(base) 분리 통계
    trainable = params.filter(train_config.trainable_filter)
    frozen = params.filter(train_config.freeze_filter)
    n_all, b_all = tree_stats(params)
    n_tr, b_tr = tree_stats(trainable)
    n_fr, b_fr = tree_stats(frozen)

    # 만약 전부 fp32였다면? (freeze→bf16 캐스팅이 아낀 정적 메모리 비교용)
    b_if_fp32 = sum(int(np.prod(l.shape)) * 4 for l in leaves(params))

    delta = (after - before) if (before >= 0 and after >= 0) else -1
    print(f"  전체 파라미터 수            : {n_all:>14,}")
    print(f"    - 동결(base, bf16)        : {n_fr:>14,}  ({mib(b_fr):8.1f} MiB)")
    print(f"    - 학습가능(LoRA 등, fp32) : {n_tr:>14,}  ({mib(b_tr):8.1f} MiB)")
    print(f"  파라미터 바이트(이론)       : {gib(b_all):6.2f} GiB  (전부 fp32였다면 {gib(b_if_fp32):.2f} GiB)")
    if delta >= 0:
        print(f"  params 실제 GPU 점유 증가 Δ : {gib(delta):6.2f} GiB")

    # ── 옵티마이저 상태 실측: 진짜 Adam(freeze=False) vs set_to_zero(우리가 바꾼 freeze=True 경로) ──
    import optax
    import openpi.training.optimizer as _optimizer
    trainable_params = params.filter(train_config.trainable_filter)

    g0 = gpu_in_use()
    tx = _optimizer.create_optimizer(train_config.optimizer, train_config.lr_schedule, weight_decay_mask=None)
    opt_state = realize(tx.init(trainable_params))   # 학습가능 파라미터마다 모멘트 m,v 2벌
    g1 = gpu_in_use()
    KEEP.append(opt_state)
    opt_delta = (g1 - g0) if (g0 >= 0 and g1 >= 0) else -1
    _, opt_b = tree_stats(opt_state)

    g2 = gpu_in_use()
    opt0_state = realize(optax.set_to_zero().init(trainable_params))  # frozen 경로 → EmptyState
    g3 = gpu_in_use()
    KEEP.append(opt0_state)
    opt0_delta = (g3 - g2) if (g2 >= 0 and g3 >= 0) else -1
    _, opt0_b = tree_stats(opt0_state)

    opt_d_str = f"{gib(opt_delta):.2f} GiB" if opt_delta >= 0 else "(CPU)"
    print(f"  Adam opt state (freeze=False): {gib(opt_b):.2f} GiB  (GPU Δ={opt_d_str})")
    print(f"    └ set_to_zero(freeze=True) : {mib(opt0_b):.1f} MiB  ← 우리가 바꾼 경로, 이만큼 절감")

    # 학습 시 상주(freeze=False): params 1벌 + target 1벌 + Adam opt
    resident_full = b_all * 2 + opt_b
    # 우리가 바꾼 freeze=True: params 1벌만(target은 참조공유, opt은 set_to_zero)
    resident_frozen = b_all + opt0_b
    ROWS.append(("pi0.5: params (bf16 base + fp32 LoRA)", n_all, b_all, delta))
    ROWS.append(("pi0.5: target_actor_params (freeze=False만)", n_all, b_all, -1))
    ROWS.append(("pi0.5: Adam opt state (freeze=False만, 실측)", n_tr * 2, opt_b, opt_delta))
    print(f"  → freeze=False 상주          : params×2 + Adam ≈ {gib(resident_full):.2f} GiB")
    print(f"  → freeze=True  상주(수정 후) : params×1            ≈ {gib(resident_frozen):.2f} GiB"
          f"   (절감 ≈ {gib(resident_full - resident_frozen):.2f} GiB)")
    return resident_full


# ──────────────────────────────────────────────────────────────────────────
# EXPO 쪽 신경망 — EXPOLearner.create()의 빌드를 그대로 복제
# ──────────────────────────────────────────────────────────────────────────
def measure_expo(args):
    from functools import partial

    from expo_ft.networks import (
        MLP, Ensemble, StateActionValue, PixelMultiplexer, PixelEditMultiplexer, BatchEncoder,
    )
    from expo_ft.networks.encoders import ResNetV2Encoder
    from expo_ft.networks.temperature import Temperature
    from expo_ft.distributions import TanhNormal
    from configs.model import expo_ft_pi_config

    mc = expo_ft_pi_config.get_config()
    latent_dim_image = int(mc.latent_dim_image)         # 512
    latent_dim_state = int(mc.latent_dim_state)         # 64
    hidden_dims = tuple(mc.hidden_dims)                 # (256,256,256)
    encoder_stage_sizes = tuple(mc.encoder_stage_sizes)  # (3,4,6,3) ← ResNet-34 토폴로지
    encoder_num_filters = int(mc.encoder_num_filters)   # 64
    include_state = bool(mc.include_state)
    critic_layer_norm = bool(mc.critic_layer_norm)
    use_pnorm = bool(mc.get("use_pnorm", False))
    actor_drop = mc.get("actor_drop", 0.0)
    num_qs = args.num_qs if args.num_qs > 0 else int(mc.num_qs)

    action_dim = args.action_dim
    state_dim = args.state_dim
    replan_steps = args.replan_steps
    full_action_dim = replan_steps * action_dim
    channels = 3 * args.num_cameras

    print(f"\n[EXPO] config: num_qs={num_qs}, latent_img={latent_dim_image}, "
          f"latent_state={latent_dim_state}, hidden={hidden_dims}, "
          f"encoder_stages={encoder_stage_sizes}x{encoder_num_filters}f")
    print(f"       dims: action_dim={action_dim}, state_dim={state_dim}, "
          f"replan_steps={replan_steps} → full_action_dim={full_action_dim}, "
          f"image={args.resize}x{args.resize}x{channels}(={args.num_cameras}cams)")

    k = jax.random.PRNGKey(0)
    ks = jax.random.split(k, 8)

    # 1) critic encoder (BatchEncoder = ResNetV2 (3,4,6,3), 9채널 단일 입력)
    encoder_cls = partial(ResNetV2Encoder, stage_sizes=encoder_stage_sizes, num_filters=encoder_num_filters)
    batch_encoder_def = BatchEncoder(encoder_cls=encoder_cls, latent_dim=latent_dim_image,
                                     pixel_keys=("pixels",), depth_keys=())
    obs_example = jnp.ones((args.resize, args.resize, channels))
    enc = measure("critic encoder (ResNetV2 batch_encoder)",
                  lambda: batch_encoder_def.init(ks[0], obs_example)["params"])

    # 2) critic ensemble (num_qs × StateActionValue MLP)
    critic_base_cls = partial(MLP, hidden_dims=hidden_dims, activate_final=True,
                              dropout_rate=None, use_layer_norm=critic_layer_norm, use_pnorm=use_pnorm)
    critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
    critic_cls = partial(Ensemble, net_cls=critic_cls, num=num_qs)
    critic_def = PixelMultiplexer(network_cls=critic_cls, latent_dim=latent_dim_state, include_state=include_state)
    critic_obs = jnp.ones((1, latent_dim_image))
    critic_actions = jnp.ones((1, full_action_dim))
    critic_states = jnp.ones((1, state_dim))
    measure(f"critic ensemble (num_qs={num_qs})",
            lambda: critic_def.init(ks[1], critic_obs, critic_actions, p=critic_states)["params"])

    # 3) residual actor (TanhNormal MLP)
    residual_actor_base_cls = partial(MLP, hidden_dims=hidden_dims, dropout_rate=actor_drop,
                                      activate_final=True, use_pnorm=use_pnorm)
    residual_actor_cls = TanhNormal(residual_actor_base_cls, full_action_dim)
    residual_actor_def = PixelEditMultiplexer(network_cls=residual_actor_cls,
                                              latent_dim=latent_dim_state, include_state=include_state)
    measure("residual actor (TanhNormal MLP)",
            lambda: residual_actor_def.init(ks[2], jnp.ones((1, latent_dim_image)),
                                            actions=jnp.ones((1, full_action_dim)), p=critic_states)["params"])

    # 4) temperature (SAC 엔트로피 온도, 스칼라)
    temp_def = Temperature(1.0)
    measure("temperature (scalar)", lambda: temp_def.init(ks[3])["params"])


# ──────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-qs", type=int, default=0, help="0이면 config 기본(10) 사용")
    ap.add_argument("--replan-steps", type=int, default=8)
    ap.add_argument("--action-dim", type=int, default=7)
    ap.add_argument("--state-dim", type=int, default=16)
    ap.add_argument("--resize", type=int, default=224)
    ap.add_argument("--num-cameras", type=int, default=3)
    ap.add_argument("--no-pi05", action="store_true")
    args = ap.parse_args()

    dev = jax.local_devices()[0]
    print("=" * 92)
    print(f"JAX device: {dev}  (platform={dev.platform})")
    lim = gpu_limit()
    if lim >= 0:
        print(f"GPU 메모리 한도(bytes_limit, MEM_FRACTION 반영): {gib(lim):.2f} GiB")
    print(f"시작 시 bytes_in_use: {mib(max(gpu_in_use(), 0)):.1f} MiB")
    print("=" * 92)

    pi05_resident = 0
    if not args.no_pi05:
        try:
            pi05_resident = measure_pi05()
        except Exception as e:
            print(f"\n[pi0.5] 측정 실패(서버 venv/openpi 필요): {type(e).__name__}: {e}")

    try:
        measure_expo(args)
    except Exception as e:
        print(f"\n[EXPO] 측정 실패: {type(e).__name__}: {e}")
        raise

    # ── 최종 표 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 92)
    print("요약: 컴포넌트별 정적(상주) 메모리")
    print("=" * 92)
    print(f"{'컴포넌트':<46}{'파라미터수':>16}{'이론바이트':>14}{'GPU Δ':>14}")
    print("-" * 92)
    tot_b = 0
    for name, n, b, d in ROWS:
        tot_b += b
        d_str = f"{mib(d):.1f} MiB" if d >= 0 else "-"
        print(f"{name:<46}{n:>16,}{mib(b):>11.1f} MiB{d_str:>14}")
    print("-" * 92)
    print(f"{'합계(이론, 위 사본 포함)':<46}{'':>16}{gib(tot_b):>11.2f} GiB")
    peak = gpu_peak()
    if peak >= 0:
        print(f"\n전체 빌드 후 GPU peak_bytes_in_use: {gib(peak):.2f} GiB")
    print("\n해석:")
    print("  · 정적 VRAM은 pi0.5(frozen base 포함)가 압도적 — frozen이라도 가중치는 GPU에 상주한다.")
    print("    freeze가 줄이는 건 옵티마이저 상태와 backward activation이지 '가중치 자체'가 아니다.")
    print("  · critic encoder/ensemble/residual actor/temperature의 정적 합은 보통 1 GiB 미만.")
    print("  · 단, 학습 update의 '순간 peak'는 위 정적 수치와 다르다. critic ResNet 인코더 forward/")
    print("    backward + pi0.5 후보 N개 샘플링 activation이 peak를 만든다(이 스크립트는 정적만 측정).")
    print("=" * 92)


if __name__ == "__main__":
    main()
