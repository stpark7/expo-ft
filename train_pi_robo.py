#! /usr/bin/env python
import os
import logging
import time
from collections import deque

import numpy as np
import tqdm
from absl import app, flags

from ml_collections import config_flags

import jax
import etils.epath as epath

import wandb
from expo_ft.agents import initialize_checkpoint_dir, save_replay_buffer_transition
from expo_ft.data.replay_buffer import create_replay_buffer
from expo_ft.data.batch_processor import BatchProcessor
from expo_ft.env.env_client import EnvClientWrapper
from expo_ft.env.droid_utils import process_droid_dataset
from expo_ft.env.robocasa_utils import process_robocasa_dataset
from expo_ft.utils.log_utils import EpisodeState, TrainingStats
from expo_ft.utils.train_utils import get_batch_info, init_logging, init_wandb

import openpi.training.sharding as openpi_sharding

import warnings
# JAX/numpy 등에서 나오는 DeprecationWarning 로그를 숨김 (기능과 무관, 출력만 정리)
warnings.filterwarnings("ignore", category=DeprecationWarning)

FLAGS = flags.FLAGS

# ──────────────────────────────────────────────────────────────────────────
# CLI 플래그 정의: 셸 스크립트(scripts/pick/*.sh)가 값을 채워서 호출하고,
# main() 안에서 FLAGS.xxx 로 접근한다. DEFINE_xxx("이름", 기본값, "설명")
# ──────────────────────────────────────────────────────────────────────────

# --- 실험/로깅 ---
flags.DEFINE_string("project_name", "expo-ft", "wandb 프로젝트 이름")
flags.DEFINE_string("run_name", None, "wandb 런 이름 (로그/체크포인트 디렉터리 이름으로도 쓰임)")
# 오프라인 배치 비율: 0이면 데모를 온라인 버퍼에 바로 삽입, >0이면 별도 오프라인 버퍼를 두고 그 비율로 섞음
flags.DEFINE_float("offline_ratio", 0.0, "Offline batch fraction; 0 inserts dataset into online replay buffer.")
flags.DEFINE_integer("seed", 42, "랜덤 시드")

# --- 학습 스케줄 (핵심) ---
# 그래디언트 업데이트를 언제 돌릴지: 에피소드마다 / 스텝마다 / 에피소드 묶음마다
flags.DEFINE_enum("update_type", "episode", ["episode", "step", "batch"], "When to run gradient updates: per episode, per step, or per batch of episodes.")
flags.DEFINE_integer("num_updates", 1, "트리거(에피소드/스텝/배치) 1회당 그래디언트 업데이트 횟수")
flags.DEFINE_integer("num_batch", 1, "update_type=batch 일 때 몇 에피소드를 모아 한 번 업데이트할지")
flags.DEFINE_integer("batch_size", 64, "미니배치 크기 (디바이스 수로 나누어떨어져야 함)")
flags.DEFINE_integer("max_steps", 100_000, "총 학습 스텝 수")
# 리플레이 버퍼 capacity를 max_steps에서 분리한다. 0이면 기존 동작(=max_steps)으로 폴백.
# 버퍼는 np.empty로 capacity만큼 잡고 채워지는 만큼 호스트 RAM이 fault된다. capacity가
# max_steps(기본 100k)에 묶여 있으면 학습이 길어질수록 온라인 버퍼가 ~58GB까지 자라고,
# 체크포인트 때 params Orbax 직렬화 호스트 복사본(~13GB)이 얹혀 62GB 머신에서 커널 OOM이 난다.
# 이 값으로 버퍼를 고정 크기 원형(circular)으로 묶으면 학습 길이와 무관하게 RAM이 bounded.
# (transition당 ≈0.59MB → 60000이면 ≈35GB. resume 시 저장된 버퍼 크기보다 크게 잡아야 안전.)
flags.DEFINE_integer("buffer_capacity", 0, "리플레이 버퍼 capacity (0 = max_steps와 동일). 호스트 RAM 상한 제어용")
flags.DEFINE_integer("num_data", 0, "로드할 오프라인 데모 에피소드 최대 개수 (0 = 전부)")
flags.DEFINE_boolean("tqdm", True, "tqdm 진행바 사용 여부")

# --- 체크포인트 ---
flags.DEFINE_boolean("checkpoint_model", False, "학습 중 에이전트 체크포인트 저장 여부")
flags.DEFINE_integer("checkpoint_interval", 0, "N 스텝마다 체크포인트 저장. 0이고 checkpoint_model=True면 마지막에만 저장")
flags.DEFINE_boolean("checkpoint_buffer", False, "리플레이 버퍼 transition까지 저장 (정확한 resume용)")
flags.DEFINE_integer("utd_ratio", 20, "Update-to-Data 비율: 배치를 미니배치로 쪼개 critic을 몇 번 스캔할지")
flags.DEFINE_integer("keep_period", None, "N 스텝마다의 체크포인트를 영구 보관")
flags.DEFINE_boolean("overwrite", False, "기존 체크포인트 디렉터리 덮어쓰기")
flags.DEFINE_boolean("resume", False, "체크포인트에서 학습 이어하기")
flags.DEFINE_string("output_dir", "./logs", "로그/체크포인트 저장 디렉터리")

# --- 분산 처리 / 서버-클라이언트 통신 ---
flags.DEFINE_integer("fsdp_devices", 1, "FSDP 샤딩에 사용할 디바이스 수")
# 서버(학습기)가 클라이언트(로봇)로 '접속해 나가는' 구조라 기본이 localhost (SSH 리버스 터널 경유)
flags.DEFINE_string("client_host", "localhost", "환경(로봇) 클라이언트 호스트")
flags.DEFINE_integer("client_port", 8102, "환경(로봇) 클라이언트 포트")

# 한 번 샘플한 액션 청크에서 실제로 실행할 스텝 수
flags.DEFINE_integer("replan_steps", 8, "Number of replan steps for evaluation.")

flags.DEFINE_string("dataset_path", "", "오프라인 데모 데이터셋 경로")

# 설정 파일 자체를 가리키는 특수 플래그 (값이 아니라 .py 설정 파일 경로를 받음)
# --config → 모델/알고리즘 설정. 내부 model_cls 문자열로 EXPOLearner / BCLearner 분기
# lock_config=False → --config.N=8 처럼 개별 스칼라 값을 CLI에서 덮어쓰기 허용
#   (단, numpy 배열 필드 bounds/reset_joints 는 CLI 오버라이드 불가 → 파일 직접 수정)
config_flags.DEFINE_config_file(
    "config",
    "configs/model/expo_ft_pi_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

# --config_task → 태스크 설정. 환경 클래스, 언어 지시문(프롬프트), 작업공간 bounds,
#   카메라 시리얼, control_hz 등을 담음. 클라이언트와 서버가 byte-identical 해야 함
config_flags.DEFINE_config_file(
    "config_task",
    "configs/task/pick.py",
    "File path to the task configuration.",
    lock_config=False,
)

def main(_):
    # ── 0. 기본 검증 & JAX 설정 ───────────────────────────────────────────
    init_logging()
    assert FLAGS.offline_ratio >= 0.0 and FLAGS.offline_ratio <= 1.0

    # batch_size는 디바이스 수로 나누어떨어져야 샤딩이 가능
    if FLAGS.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {FLAGS.batch_size} must be divisible by "
            f"the number of devices {jax.device_count()}"
        )
    # JAX 컴파일 결과를 디스크에 캐시 → 재실행 시 컴파일 시간 단축
    jax.config.update(
        "jax_compilation_cache_dir",
        str(epath.Path("~/.cache/jax").expanduser()),
    )

    # FSDP mesh와 샤딩 정의: data_sharding=배치 축 분할, replicated=모든 디바이스에 복제
    # !
    mesh = openpi_sharding.make_mesh(FLAGS.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(openpi_sharding.DATA_AXIS)
    )
    replicated_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec()
    )

    # ── 1. 로그/체크포인트 디렉터리 & wandb ──────────────────────────────
    log_dir = os.path.join(FLAGS.output_dir, FLAGS.run_name)
    os.makedirs(log_dir, exist_ok=True)
    train_video_dir = os.path.join(log_dir, "train_videos")
    os.makedirs(train_video_dir, exist_ok=True)
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Orbax 체크포인트 매니저 준비. resume/overwrite 플래그를 반영하고,
    # 실제로 이어서 학습하는 경우 resuming=True 를 돌려줌 (이후 복원 분기의 기준)
    checkpoint_dir_path = epath.Path(checkpoint_dir)
    checkpoint_manager, resuming = initialize_checkpoint_dir(
        checkpoint_dir_path,
        keep_period=FLAGS.keep_period,
        overwrite=FLAGS.overwrite,
        resume=FLAGS.resume,
    )

    init_wandb(checkpoint_dir_path, resuming, FLAGS.project_name, FLAGS.run_name)
    wandb.config.update(FLAGS.flag_values_dict(), allow_val_change=resuming)

    # ── 2. 오프라인 데모 데이터셋 로드 ───────────────────────────────────
    # HDF5 데모를 transition dict 리스트로 변환. example_action 은 shape 참조용
    if FLAGS.config_task.env_type == 'droid':
        dataset = process_droid_dataset(
            FLAGS.dataset_path,
            FLAGS.config_task,
            num_data=FLAGS.num_data,
        )
    elif FLAGS.config_task.env_type == 'sim':
        # RoboCasa365 LeRobot 데모 -> arm 7차원 action + 16차원 state transition.
        dataset = process_robocasa_dataset(
            FLAGS.dataset_path,
            FLAGS.config_task,
            num_data=FLAGS.num_data,
        )
    else:
        raise ValueError(f"Unsupported dataset type: {FLAGS.config_task.env_type}")
    example_action = dataset[0]['actions'][np.newaxis]

    # ── 3. 환경(로봇) 래퍼 생성 ──────────────────────────────────────────
    # 실제 하드웨어는 건드리지 않고, WebSocket RPC로 클라이언트의 DroidEnv를 원격 제어
    train_env_creation_request = {
        "example_action": example_action,
        "env_usage": "train",
        "video_dir": train_video_dir,
    }

    logging.info("Creating environment...")
    env = EnvClientWrapper(
        env_creation_request=train_env_creation_request,
        host=FLAGS.client_host,
        port=FLAGS.client_port
    )
    env.reset()
    logging.info(f"Created training environment {env.env_id}")

    # ── 4. 알고리즘 디스패치 (레지스트리 없이 if/elif) ──────────────────
    # config.model_cls 문자열로 어떤 학습기를 쓸지 결정
    model_cls = FLAGS.config.model_cls
    # BCLearner(DAgger baseline)는 critic 없이 사람 개입 청크만으로 actor를 학습
    use_dagger_hil_sampling = model_cls == "BCLearner"
    if model_cls == "BCLearner":
        from expo_ft.agents.alg.bc import load_agent, restore_checkpoint, save_checkpoint
    elif model_cls == "EXPOLearner":
        from expo_ft.agents.alg.expo_ft import load_agent, restore_checkpoint, save_checkpoint
    else:
        raise ValueError(f"Unsupported model class: {model_cls}")

    # pi0.5 VLA 로드: actor 네트워크, train state, EMA target params, 샤딩 정보
    from expo_ft.agents.vla.pi05 import build_pi05
    actor, actor_train_state, target_actor_params, agent_kwargs, vla_metadata = build_pi05(
        FLAGS.config, FLAGS.seed, mesh, data_sharding, replicated_sharding,
        resuming, env.task_description,
    )

    # ── 5. 리플레이 버퍼 2개 (온라인 + 오프라인) ────────────────────────
    # 버퍼 capacity는 max_steps와 분리(buffer_capacity>0이면 그 값). 학습 길이가 길어도
    # 호스트 RAM이 bounded가 되도록 고정 크기 원형 버퍼로 묶는다. 0이면 기존대로 max_steps.
    rb_capacity = FLAGS.buffer_capacity if FLAGS.buffer_capacity > 0 else FLAGS.max_steps
    logging.info(
        "Replay buffer capacity = %d (buffer_capacity=%d, max_steps=%d)",
        rb_capacity, FLAGS.buffer_capacity, FLAGS.max_steps,
    )
    rb_args = dict(
        config=FLAGS.config,
        example_action=example_action,
        capacity=rb_capacity,
        task_description=env.task_description,
        replan_steps=FLAGS.replan_steps,
        seed=FLAGS.seed,
    )
    replay_buffer = create_replay_buffer(**rb_args)          # 온라인: 로봇이 모은 transition
    offline_replay_buffer = create_replay_buffer(**rb_args)  # 오프라인: 사전 데모 (offline_ratio>0일 때)

    # critic 배치 + actor 배치를 조립하는 오케스트레이터
    # actor_success_only: EXPO는 기본적으로 '성공' transition만으로 base actor를 학습
    actor_success_only = getattr(FLAGS.config, "actor_success_only", False)
    batch_processor = BatchProcessor(
        replay_buffer=replay_buffer,
        offline_replay_buffer=offline_replay_buffer,
        data_sharding=data_sharding,
        batch_size=FLAGS.batch_size,
        utd_ratio=FLAGS.utd_ratio,
        offline_ratio=FLAGS.offline_ratio,
        actor_success_only=actor_success_only,
        use_dagger_hil_sampling=use_dagger_hil_sampling,
        dataset=dataset,
    )

    # ── 6. 예시 샘플로 실제 action/state 차원 확정 후 에이전트 생성 ─────
    # critic이 관측에 쓸 카메라 키들. DROID는 (base, left_wrist) 2대 기본,
    # robocasa는 task config에서 (base, left_wrist, right_wrist) 3대를 명시.
    critic_camera_keys = tuple(
        FLAGS.config_task.get("critic_camera_keys", ("base_0_rgb", "left_wrist_0_rgb"))
    )
    # 버퍼에서 샘플 하나를 critic 포맷으로 변환해 네트워크 초기화에 쓸 shape를 얻음.
    # critic_camera_keys에 right_wrist가 있으면 예시도 3대(9채널)로 만들어 인코더
    # 입력 채널 수가 그에 맞게 초기화되게 한다.
    example_inputs = {
        "base_image": offline_replay_buffer.dataset_dict['base_image'][0][np.newaxis],
        "left_wrist_image": offline_replay_buffer.dataset_dict['left_wrist_image'][0][np.newaxis],
        "state": offline_replay_buffer.dataset_dict['state'][0][np.newaxis],
        "actions": offline_replay_buffer.dataset_dict['actions'][0][np.newaxis],
    }
    if "right_wrist_0_rgb" in critic_camera_keys:
        example_inputs["right_wrist_image"] = offline_replay_buffer.dataset_dict['right_wrist_image'][0][np.newaxis]
    agent_example_observation, agent_example_state, agent_example_action = offline_replay_buffer.convert_to_critic_format(
        example_inputs
    )
    actor.action_dim = agent_example_action.squeeze().shape[-1]
    actor.state_dim = agent_example_state.squeeze().shape[-1]
    # arm(action_dim)을 모델 32차원의 어디에 놓을지(=offset). 현재 모든 태스크 0(arm-first).
    actor.action_pad_offset = FLAGS.config_task.get("action_pad_offset", 0)
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
        resume=resuming,
        replan_steps=FLAGS.replan_steps,
        default_prompt=env.task_description,
        residual_action_xyzg=FLAGS.config_task.residual_action_xyzg,
        critic_camera_keys=critic_camera_keys,
    )
    
    # ── 7. resume 시 체크포인트에서 에이전트 & 버퍼 복원 ────────────────
    start_step = 0
    if resuming:
        agent = restore_checkpoint(checkpoint_manager, agent)
        agent = agent.cache_infer_params()  # 추론 파라미터를 한 디바이스에 고정 (롤아웃마다 device_put 회피)
        steps = tuple(checkpoint_manager.all_steps())
        latest_step = max(steps) if steps else None
        if latest_step is not None:
            start_step = latest_step
            logging.info("Resuming from step %d", start_step)
        batch_processor.restore(checkpoint_dir_path, up_to_step=latest_step)

    # ── 8. 로그 상태 & 에피소드 초기화 ───────────────────────────────────
    episode_log = EpisodeState()
    training_log = TrainingStats(
        ep_count=replay_buffer.count_episodes_chronological() if resuming else 0,
    )
    logging.info("Resuming: ep_count set to %d (episodes in replay buffer).", training_log.ep_count)

    batch_processor.on_episode_start()

    # ── 9. 메인 루프 준비 ────────────────────────────────────────────────
    dt = 1.0 / FLAGS.config_task.control_hz  # 제어 주기(초). 기본 10Hz → 0.1s
    done = False
    env.reset()
    start_step_time = time.time()
    env.step(FLAGS.config_task.example_action.squeeze().tolist())  # 첫 더미 스텝으로 파이프라인 워밍업
    action_plan = deque()       # 샘플한 액션 청크를 담는 큐 (popleft로 한 스텝씩 소비)
    action_type = "policy"      # "policy"=정책 실행, "human"=사람 개입 중
    episodes_since_update = 0   # update_type=batch 일 때 누적 에피소드 카운터
    combine_rng = jax.random.PRNGKey(FLAGS.seed + 100)

    # 그래디언트 업데이트 묶음. nonlocal로 바깥 스코프의 agent/rng를 갱신
    def run_agent_updates(num_updates: int, metrics: dict):
        nonlocal agent, combine_rng
        for _ in range(num_updates):
            update_start = time.time()
            # critic 배치 + actor 배치를 뽑음 (offline/online 혼합, success-only 필터 적용)
            batch, actor_batch, combine_rng = batch_processor.next_batch(combine_rng)
            metrics["batch_info"] = get_batch_info(batch)
            agent = agent.replace(rng=jax.device_put(agent.rng, replicated_sharding))
            # 한 번의 update에서 critic / residual actor+temperature / pi0.5 base actor를 모두 학습
            agent, update_info = agent.update(agent, batch, FLAGS.utd_ratio, actor_batch)
            training_log.record_update_time(time.time() - update_start, metrics)
            for k, v in update_info.items():
                metrics[f"training/{k}"] = v

    # ── 10. 메인 제어 루프: 매 스텝 관측 → 샘플 → 실행 → 저장 → 업데이트 ─
    for i in tqdm.tqdm(
        range(start_step, FLAGS.max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm
    ):
        loop_start = time.time()
        step_metrics = {}

        # (a) 현재 관측과 직전 스텝의 결과(done/success/reward/mask) 수신
        observation = env.get_observation()
        done, success, reward, mask = env.get_info_for_step()

        # (b) 액션 샘플링: 큐가 비었고 사람이 조종 중이 아닐 때만 정책 추론
        #     sample_actions = pi0.5 후보 N개 propose → residual edit → Q critic으로 select
        if not action_plan and action_type != "human":
            sample_start = time.time()
            action_chunk, agent, new_si = agent.sample_actions(observation)
            episode_log.sample_info_history.append(new_si)
            training_log.record_sample_time(time.time() - sample_start, step_metrics)
            action_plan.extend(action_chunk[:FLAGS.replan_steps])  # 청크에서 replan_steps개만 큐에 적재
        else:
            # 사람 개입 중이거나 큐에 액션이 남아있으면 추론 생략 (직전 sample_info 재사용)
            episode_log.sample_info_history.append(episode_log.sample_info_history[-1] if episode_log.sample_info_history else None)

        # (c) 제어 주기 유지: dt보다 빨리 끝났으면 남는 시간만큼 대기
        elapsed = time.time() - start_step_time
        if elapsed < dt:
            time.sleep(dt - elapsed)

        # (d) 큐에서 액션 하나를 꺼내 로봇 실행. real_action=실제 적용된 액션, action_type=policy/human
        has_action = bool(action_plan)
        action = action_plan.popleft() if has_action else np.zeros_like(example_action.squeeze())
        real_action, action_type = env.step(action.tolist())
        start_step_time = time.time()

        episode_log.record_step(observation, len(action_plan), action_type, real_action, reward)

        # 사람이 개입을 시작하면 정책이 만든 남은 액션 청크는 폐기
        if action_type == "human":
            action_plan.clear()

        # (e) transition을 리플레이 버퍼에 저장 (정책 실행 또는 사람 개입 스텝만)
        if has_action or action_type == "human":
            transition_dict = dict(
                observations=observation,
                actions=real_action,
                rewards=reward,
                masks=mask,
                dones=done,
                is_hil=(action_type == "human"),  # 사람 개입 여부 (BC/DAgger 학습 필터에 쓰임)
            )
            batch_processor.insert_transition(transition_dict)

        # (f) 업데이트 트리거: 에피소드 10개 이상 + 스텝이 batch_size 이상 쌓여야 시작
        can_update = training_log.ep_count >= 10 and i >= FLAGS.batch_size
        if FLAGS.update_type == "step" and can_update:
            run_agent_updates(FLAGS.num_updates, step_metrics)

        # (g) 에피소드 종료 처리
        if done:
            batch_processor.on_episode_done(success)
            env.reset()

            # update_type에 따라 에피소드 종료 시점에 업데이트 (episode=매번, batch=num_batch마다)
            if FLAGS.update_type == "episode" and can_update:
                for _ in tqdm.tqdm(range(FLAGS.num_updates)):
                    run_agent_updates(1, step_metrics)
            elif FLAGS.update_type == "batch" and can_update:
                episodes_since_update += 1
                if episodes_since_update >= FLAGS.num_batch:
                    for _ in tqdm.tqdm(range(FLAGS.num_updates)):
                        run_agent_updates(1, step_metrics)
                    episodes_since_update = 0

            # 로그 정리 후 다음 에피소드 시작 상태로 리셋
            training_log.on_episode_done(episode_log, success, step_metrics)
            episode_log.reset()
            batch_processor.on_episode_start()

            observation = env.get_observation()
            done = False
            action_type = "policy"
            action_plan.clear()

        # (h) 주기적 체크포인트 저장 (checkpoint_interval 마다)
        if FLAGS.checkpoint_model and FLAGS.checkpoint_interval > 0 and i > 0 and i % FLAGS.checkpoint_interval == 0:
            try:
                save_checkpoint(checkpoint_manager, agent, i)
                logging.info(f"Saved agent checkpoint at step {i} (interval={FLAGS.checkpoint_interval})")
            except Exception as e:
                logging.error(f"Could not save model checkpoint: {e}")

        # 버퍼 transition도 저장하면 정확한 resume 가능
        if FLAGS.checkpoint_buffer and (has_action or action_type == "human"):
            try:
                save_replay_buffer_transition(checkpoint_dir_path, transition_dict, step=i)
            except Exception:
                logging.exception("Could not save agent buffer.")

        step_metrics["training/loop_time_ms"] = (time.time() - loop_start) * 1000.0
        wandb.log(step_metrics, step=i)

    # ── 11. 종료: 최종 체크포인트 저장 ───────────────────────────────────
    if FLAGS.checkpoint_model:
        try:
            save_checkpoint(checkpoint_manager, agent, FLAGS.max_steps)
            logging.info(f"Saved final agent checkpoint at step {FLAGS.max_steps}")
        except Exception as e:
            logging.error(f"Could not save final checkpoint: {e}")
        logging.info("Waiting for checkpoint manager to finish")
        checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    app.run(main)