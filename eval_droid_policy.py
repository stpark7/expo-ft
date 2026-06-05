#!/usr/bin/env python
"""정책 평가(rollout 전용) 스크립트.

학습된 체크포인트를 복원해 환경에서 에피소드를 굴리고 성공률만 집계한다.
train_pi_robo.py 와 달리 그래디언트 업데이트도, 리플레이 버퍼 저장도 없다
(리플레이 버퍼는 오직 obs/state/action shape 추출용으로만 잠깐 쓴다).

파일 이름은 droid 이지만 env_type 분기로 droid(실로봇)와 sim(RoboCasa365)을 모두 평가한다.
서버(여기)는 하드웨어를 직접 만지지 않고, WebSocket RPC로 client/run_client.py의
실제 환경을 원격 제어한다(reset/step/get_observation/get_info_for_step).

호출은 scripts/{pick,sim}/eval_policy.sh 가 플래그를 채워서 한다.
"""

from __future__ import annotations

import logging
import os
import time
from collections import deque

import etils.epath as epath
import jax
import numpy as np
from absl import app, flags
from ml_collections import config_flags

from expo_ft.agents import initialize_checkpoint_dir
from expo_ft.data.replay_buffer import create_replay_buffer
from expo_ft.env.env_client import EnvClientWrapper
from expo_ft.env.droid_utils import process_droid_dataset
from expo_ft.env.robocasa_utils import process_robocasa_dataset

import openpi.training.sharding as openpi_sharding

# ──────────────────────────────────────────────────────────────────────────
# CLI 플래그 정의: scripts/{pick,sim}/eval_policy.sh 가 값을 채워 호출하고,
# main() 안에서 FLAGS.xxx 로 접근한다. 평가는 학습과 동일한 config/체크포인트를
# 써야 하므로, 아래 값들은 학습 때 쓴 설정과 반드시 일치해야 한다.
# ──────────────────────────────────────────────────────────────────────────

# --config → 모델/알고리즘 설정. 내부 model_cls 로 EXPOLearner / BCLearner 분기.
#   ⚠️ 체크포인트를 만든 학습 config와 동일해야 네트워크 구조가 맞아 복원이 된다.
config_flags.DEFINE_config_file(
    "config",
    "configs/model/expo_ft_pi_config.py",
    "Training config (must match the checkpoint).",
    lock_config=False,
)
# --config_task → 태스크 설정. env_type(droid/sim), 언어 지시문(프롬프트),
#   control_hz, 에피소드 길이 상한 등을 담음. 클라이언트와 byte-identical 해야 함.
config_flags.DEFINE_config_file(
    "config_task",
    "configs/task/pick.py",
    "Task config (must match training).",
    lock_config=False,
)

FLAGS = flags.FLAGS
# 데모 데이터셋 경로 — 평가는 데모로 학습하지 않고, 단 1개 샘플을 transform에 통과시켜
#   action_dim/state_dim 등 shape를 도출하는 용도로만 쓴다(아래 example_action 참고).
flags.DEFINE_string("dataset_path", "", "Path to DROID dataset (for example_action).")
flags.DEFINE_integer("num_data", 1, "Number of episodes to load from dataset (only need 1 for example_action).")
flags.DEFINE_integer("seed", 42, "Random seed.")
# 체크포인트 디렉터리와 불러올 step. step 미지정 시 가장 최신 step을 자동 선택.
flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory (e.g. .../checkpoints/<run_name>/checkpoints).")
flags.DEFINE_integer("checkpoint_step", None, "Checkpoint step to load; default is latest.")
# 환경(로봇/sim) 클라이언트 접속 정보. 서버가 클라이언트로 '접속해 나가는' 구조라 기본 localhost.
flags.DEFINE_string("client_host", "localhost", "Rollout server host.")
flags.DEFINE_integer("client_port", 8102, "Rollout server port.")
flags.DEFINE_integer("num_episodes", 10, "Number of evaluation episodes.")
# 한 번 샘플한 액션 청크에서 실제 실행할 스텝 수. 학습 때 값과 맞춰야 분포가 동일.
flags.DEFINE_integer("replan_steps", 8, "Replan every N steps (match training).")
# True면 residual/critic 없이 pi0.5 base 정책 액션만 사용(후보 1개) → 베이스 성능 측정용.
flags.DEFINE_boolean("only_base_actions", False, "Use only base (OpenPI) actions, no residual, sample 1.")
flags.DEFINE_boolean("save_video", True, "Save evaluation videos.")
flags.DEFINE_integer("fsdp_devices", 1, "Number of FSDP devices (match training).")


def main(_):
    # ── 0. 설정 로드 & 기본 검증 ──────────────────────────────────────────
    config = FLAGS.config              # 모델/알고리즘 설정 (--config 가 가리키는 .py)
    config_task = FLAGS.config_task    # 태스크 설정 (--config_task 가 가리키는 .py)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger(__name__)

    # 이 스크립트는 실로봇(droid)과 시뮬레이션(sim) 두 환경만 평가한다.
    if config_task.env_type not in ("droid", "sim"):
        raise ValueError("config_task.env_type must be 'droid' or 'sim'.")

    if not FLAGS.dataset_path or not FLAGS.checkpoint_dir:
        raise ValueError("--dataset_path and --checkpoint_dir are required.")

    # ── 1. 체크포인트 매니저 열고 불러올 step 결정 ───────────────────────
    checkpoint_dir_path = epath.Path(FLAGS.checkpoint_dir)
    if not checkpoint_dir_path.exists():
        raise FileNotFoundError(f"Checkpoint dir not found: {checkpoint_dir_path}")
    # resume=True 로 기존 체크포인트 디렉터리를 '읽기' 모드로 연다(평가는 새로 쓰지 않음).
    checkpoint_manager, _ = initialize_checkpoint_dir(
        checkpoint_dir_path,
        keep_period=None,
        overwrite=False,
        resume=True,
    )
    checkpoint_steps = tuple(checkpoint_manager.all_steps())
    step = FLAGS.checkpoint_step
    # step 미지정이면 가장 최신 체크포인트를 사용. (하나도 없으면 0 = 미복원 = 무작위 초기 가중치)
    if step is None:
        step = max(checkpoint_steps) if checkpoint_steps else 0
        logger.info("Using latest checkpoint step %s", step)
    # step!=0 이면 실제 복원할 step이므로, 디렉터리에 존재하는지 확인.
    if step != 0:
        if step not in checkpoint_steps:
            raise ValueError(f"Step {step} not in checkpoint steps {checkpoint_steps}")
        logger.info("Will load checkpoint at step %s", step)

    # ── 2. 데모 1개 로드 → shape 도출 & 에피소드 길이 상한 결정 ──────────
    # 데모는 학습이 아니라 obs/action transform을 한 번 통과시켜 shape를 얻는 용도.
    # example_action(=action_dim)은 config가 아니라 dataset에서 도출한다.
    if config_task.env_type == "droid":
        dataset = process_droid_dataset(
            FLAGS.dataset_path, config_task, num_data=FLAGS.num_data,
        )
        max_traj_len = config_task.auto_reset_steps  # droid: 자동 리셋까지의 스텝 수
    else:  # sim (robocasa)
        dataset = process_robocasa_dataset(
            FLAGS.dataset_path, config_task, num_data=FLAGS.num_data,
        )
        max_traj_len = config_task.max_steps             # sim: done 미보고이므로 이 상한으로 끊음
    example_action = dataset[0]["actions"][np.newaxis]   # [1, action_dim] — env 워밍업/shape 참조용

    task_description = config_task.language_instruction  # VLA 프롬프트로 그대로 사용
    dt = 1.0 / config_task.control_hz                    # 제어 주기(초). 한 스텝의 목표 소요시간

    # ── 3. JAX mesh/샤딩 & 알고리즘별 로더 선택 (train_pi_robo와 동일) ────
    mesh = openpi_sharding.make_mesh(FLAGS.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(openpi_sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # config.model_cls 문자열로 EXPO(critic+residual) / BC(베이스라인) 로더를 분기.
    model_cls = config.model_cls
    if model_cls == "BCLearner":
        from expo_ft.agents.alg.bc import load_agent, restore_checkpoint
    elif model_cls == "EXPOLearner":
        from expo_ft.agents.alg.expo_ft import load_agent, restore_checkpoint
    else:
        raise ValueError(f"Unsupported model class: {model_cls}")

    # ── 4. pi0.5 VLA 백본 빌드 ──────────────────────────────────────────
    # actor(정책망), actor_train_state(옵티마이저 상태), target_actor_params(EMA),
    # agent_kwargs/vla_metadata(transform·norm stat 등)를 OpenPI 포크에서 로드.
    from expo_ft.agents.vla.pi05 import build_pi05
    actor, actor_train_state, target_actor_params, agent_kwargs, vla_metadata = build_pi05(
        config, FLAGS.seed, mesh, data_sharding, replicated_sharding,
        resume=(step != 0), default_prompt=task_description,
    )

    # ── 5. 리플레이 버퍼로 agent용 example shape 추출 ────────────────────
    # 학습이 아니라, 데모 1개를 버퍼에 넣고 critic 입력 포맷으로 변환해
    # observation/state/action의 정확한 shape를 뽑아내기 위한 용도.
    replay_buffer = create_replay_buffer(
        config=config,
        example_action=example_action,
        capacity=max_traj_len * 2,
        task_description=task_description,
        replan_steps=FLAGS.replan_steps,
        seed=FLAGS.seed,
    )
    replay_buffer.insert_dataset(dataset[:1])

    # 데모 첫 transition을 critic 입력 포맷으로 변환해 example shape를 얻는다.
    agent_example_observation, agent_example_state, agent_example_action = replay_buffer.convert_to_critic_format({
        "base_image": replay_buffer.dataset_dict["base_image"][0][np.newaxis],
        "left_wrist_image": replay_buffer.dataset_dict["left_wrist_image"][0][np.newaxis],
        "state": replay_buffer.dataset_dict["state"][0][np.newaxis],
        "actions": replay_buffer.dataset_dict["actions"][0][np.newaxis],
    })
    # actor의 action/state 차원을 '데이터에서 도출한' 실제 폭으로 확정(체크포인트와 일치해야 함).
    actor.action_dim = agent_example_action.squeeze().shape[-1]
    actor.state_dim = agent_example_state.squeeze().shape[-1]
    # arm(action_dim)을 모델 32차원의 어디에 놓을지(=offset). 현재 모든 태스크 0(arm-first).
    actor.action_pad_offset = config_task.get("action_pad_offset", 0)
    # ── 6. agent 조립 → 체크포인트 복원 → 추론 파라미터 캐싱 ───────────────
    agent = load_agent(
        seed=FLAGS.seed,
        example_observation=agent_example_observation.squeeze(),
        example_action=agent_example_action.squeeze(),
        example_state=agent_example_state.squeeze(),
        actor=actor,
        actor_train_state=actor_train_state,
        target_actor_params=target_actor_params,
        agent_kwargs=agent_kwargs,
        metadata=vla_metadata,
        mesh=mesh,
        data_sharding=data_sharding,
        replicated_sharding=replicated_sharding,
        resume=(step != 0),
        replan_steps=FLAGS.replan_steps,
        default_prompt=task_description,
        residual_action_xyzg=config_task.residual_action_xyzg,
    )

    # step!=0 일 때만 실제 학습 가중치를 복원(0이면 빌드된 초기 가중치 그대로 평가).
    if step != 0:
        agent = restore_checkpoint(checkpoint_manager, agent, step=step)
        logger.info("Loaded checkpoint at step %s", step)

    # 추론 파라미터를 한 디바이스에 고정(rollout 매 스텝의 device_put 비용 제거).
    if hasattr(agent, 'cache_infer_params'):
        agent = agent.cache_infer_params()

    # ── 7. 평가 비디오 저장 경로 준비 ────────────────────────────────────
    # <checkpoint_dir>의 부모/eval/step_<step>/{full|only_base} 아래에 저장.
    video_dir = None
    if FLAGS.save_video:
        save_video_dir = os.path.join(
            os.path.dirname(FLAGS.checkpoint_dir), "eval", f"step_{step}"
        )
        parts = []
        if FLAGS.only_base_actions:
            parts.append("only_base")  # 베이스 정책만 평가한 결과는 별도 폴더로 구분
        subdir = "_".join(parts) if parts else "full"
        video_dir = os.path.join(save_video_dir, subdir)
        os.makedirs(video_dir, exist_ok=True)
        logger.info("Saving evaluation videos to %s", video_dir)

    # ── 8. 환경 클라이언트 연결 (WebSocket RPC) ──────────────────────────
    # env_usage="eval" 로 클라이언트에 평가용 환경 생성을 요청. 이후 reset/step/
    # get_observation/get_info_for_step 호출은 모두 네트워크 너머의 실제 환경으로 전달된다.
    eval_env_creation_request = {
        "example_action": example_action,
        "env_usage": "eval",
        "video_dir": video_dir or "",
    }
    logger.info("Connecting to rollout server at %s:%s ...", FLAGS.client_host, FLAGS.client_port)
    env = EnvClientWrapper(
        env_creation_request=eval_env_creation_request,
        host=FLAGS.client_host,
        port=FLAGS.client_port,
    )
    print("resetting environment...")
    env.reset()
    print("environment reset")

    time.sleep(10)  # 실로봇 하드웨어 안정화 대기(첫 reset 직후). sim에선 불필요한 지연.

    # ── 9. 평가 루프: num_episodes 만큼 에피소드를 굴리고 성공/리턴/길이 집계 ─
    successes = []        # 에피소드별 성공 여부
    episode_returns = []  # 에피소드별 누적 보상
    episode_lengths = []  # 에피소드별 스텝 수
    episode_prompts = []  # 에피소드별 언어 지시문(대상 객체) — 객체별 성공 분해용

    for ep in range(FLAGS.num_episodes):
        # 재현평가: sim은 에피소드별 고정 seed(=base_seed+ep)로 reset해, 실행마다
        # 동일한 장면·대상객체·언어 지시문 세트를 평가한다(객체가 무작위로 매번 바뀌면
        # 성공률이 실행마다 달라지고 어떤 객체가 나왔는지도 통제 못함). base_seed는
        # FLAGS.seed를 재사용한다. droid(실로봇)는 시드로 장면을 고정할 수 없어 None.
        reset_seed = (FLAGS.seed + ep) if config_task.env_type == "sim" else None
        logger.info("Episode %d / %d (reset_seed=%s)", ep + 1, FLAGS.num_episodes, reset_seed)
        observation = env.reset(seed=reset_seed)
        # 이 에피소드에 sim이 샘플한 언어 지시문(=대상 객체). VLA가 실제로 받는 프롬프트이며,
        # 동일 seed면 실행마다 같아야 한다. 객체별 성공/실패를 사후에 분해할 수 있게 기록한다.
        ep_prompt = observation.get("prompt", "") if isinstance(observation, dict) else ""
        logger.info("  prompt: %r", ep_prompt)
        start_time = time.time()
        action_plan = deque()       # 샘플한 액션 청크를 담아 한 스텝씩 꺼내 실행하는 큐
        sample_info_history = []
        ep_return = 0.0
        ep_len = 0

        # max_traj_len 은 done(성공)이 안 떠도 에피소드를 강제 종료하는 스텝 상한.
        for step in range(max_traj_len):
            step_t0 = time.time()
            # 단계별 소요시간(ms) 측정용 — 정책 추론이 제어 주기를 못 따라가는지 진단에 쓴다.
            timing = {
                "wait_ms": 0.0,
                "obs_ms": 0.0,
                "info_ms": 0.0,
                "plan_ms": 0.0,
                "act_ms": 0.0,
            }

            # (a) 현재 관측 수신
            t_obs0 = time.time()
            observation = env.get_observation()
            timing["obs_ms"] = (time.time() - t_obs0) * 1000.0
            # (b) 직전 스텝의 결과(done/success/reward) 수신. 첫 스텝은 reset 직후 상태.
            t_info0 = time.time()
            done, success, reward, _ = env.get_info_for_step()
            timing["info_ms"] = (time.time() - t_info0) * 1000.0

            # (c) 큐가 비었을 때만 정책 추론 → 청크에서 replan_steps개만 큐에 적재.
            #     only_base_actions=True면 residual/critic 없이 pi0.5 base 액션만 사용.
            t_plan0 = time.time()
            if not action_plan:
                action_chunk, agent, new_si = agent.sample_actions(
                    observation,
                    only_base_actions=FLAGS.only_base_actions,
                )
                action_chunk = np.asarray(jax.device_get(action_chunk))  # JAX→numpy (host로 가져옴)
                if action_chunk.ndim == 1:
                    action_chunk = action_chunk[None, :]  # 단일 액션이면 [1, dim]으로 보정
                action_plan.extend(list(action_chunk[: FLAGS.replan_steps]))
                sample_info_history.append(new_si)
            else:
                # 큐에 액션이 남아있으면 추론 생략(직전 sample_info 재사용).
                sample_info_history.append(sample_info_history[-1] if sample_info_history else None)
            timing["plan_ms"] = (time.time() - t_plan0) * 1000.0
            action = action_plan.popleft()  # 이번 스텝에 실행할 액션 하나

            ep_return += reward
            ep_len += 1

            # (d) 직전 스텝에서 done이 떴으면(=성공/종료) 실행하지 않고 에피소드 종료.
            if done:
                timing_total_ms = (time.time() - step_t0) * 1000.0
                logger.info(
                    "[timing][ep %d step %d] total=%.1fms wait=%.1f obs=%.1f info=%.1f plan=%.1f act=%.1f done=%s",
                    ep + 1,
                    step,
                    timing_total_ms,
                    timing["wait_ms"],
                    timing["obs_ms"],
                    timing["info_ms"],
                    timing["plan_ms"],
                    timing["act_ms"],
                    done,
                )
                break

            # (e) 제어 주기(dt) 유지: 직전 스텝 종료 후 dt가 안 지났으면 남는 만큼 대기.
            #     DROID 수집 루프와 동일한 타이밍을 맞추기 위함(실시간 페이싱).
            elapsed = time.time() - start_time
            sleep_left = dt - elapsed
            if sleep_left > 0:
                t_wait0 = time.time()
                time.sleep(sleep_left)
                timing["wait_ms"] = (time.time() - t_wait0) * 1000.0

            # (f) 액션을 환경에 실행(WebSocket RPC). 다음 스텝의 dt 기준점을 갱신.
            t_act0 = time.time()
            env.step(np.asarray(action).tolist())
            timing["act_ms"] = (time.time() - t_act0) * 1000.0
            start_time = time.time()

            timing_total_ms = (time.time() - step_t0) * 1000.0
            logger.info(
                "[timing][ep %d step %d] total=%.1fms wait=%.1f obs=%.1f info=%.1f plan=%.1f act=%.1f done=%s",
                ep + 1,
                step,
                timing_total_ms,
                timing["wait_ms"],
                timing["obs_ms"],
                timing["info_ms"],
                timing["plan_ms"],
                timing["act_ms"],
                done,
            )

        # 에피소드 종료(성공으로 break 했거나 스텝 상한 도달) → 결과 누적.
        successes.append(success)
        episode_returns.append(ep_return)
        episode_lengths.append(ep_len)
        episode_prompts.append(ep_prompt)
        logger.info("  success=%s return=%.1f len=%d prompt=%r", success, ep_return, ep_len, ep_prompt)

    # ── 10. 전체 집계 출력: 성공률 / 평균 리턴 / 평균 길이 ───────────────
    n = len(successes)
    success_rate = float(np.mean(successes))
    mean_return = float(np.mean(episode_returns))
    mean_len = float(np.mean(episode_lengths))
    logger.info("Evaluation complete: success_rate=%.2f (%d/%d) mean_return=%.2f mean_len=%.1f",
                success_rate, int(np.sum(successes)), n, mean_return, mean_len)

    # 지시문(대상 객체)별 성공 분해 — sim에서 에피소드마다 다른 객체가 나오므로,
    # 어떤 객체에서 성공/실패했는지 집계해 단일 성공률 뒤에 가려진 편차를 드러낸다.
    if any(p for p in episode_prompts):
        per_prompt: dict = {}
        for p, s in zip(episode_prompts, successes):
            hit, tot = per_prompt.get(p, (0, 0))
            per_prompt[p] = (hit + int(bool(s)), tot + 1)
        logger.info("Per-instruction success (%d distinct):", len(per_prompt))
        for p in sorted(per_prompt):
            hit, tot = per_prompt[p]
            logger.info("  [%d/%d] %r", hit, tot, p)

    print(f"success_rate={success_rate:.2f} mean_return={mean_return:.2f} mean_len={mean_len:.1f}")


if __name__ == "__main__":
    app.run(main)
