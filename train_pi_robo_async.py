#! /usr/bin/env python
import os
import logging
import time
import threading
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
from expo_ft.utils.eval_utils import TrainingEvaluator

import openpi.training.sharding as openpi_sharding

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

FLAGS = flags.FLAGS

flags.DEFINE_string("project_name", "expo-ft", "wandb project name.")
flags.DEFINE_string("run_name", None, "Optional wandb run name.")
flags.DEFINE_float("offline_ratio", 0.0, "Offline batch fraction; 0 inserts dataset into online replay buffer.")
flags.DEFINE_boolean("truncate_offline_at_success", False, "RoboCasa 데모를 첫 성공 프레임에서 잘라 done=1로(단일-terminal 보상). 온라인과 n-step 타깃 일관성(DICE-RL ph_finetune식).")
flags.DEFINE_integer("seed", 42, "Random seed.")
# ≥0이면 매 에피소드 이 seed로 sim 장면(레이아웃·스타일·객체·배치·로봇포즈·언어지시문)을
# 단일 고정 환경으로 묶는다(환경 다양성이 학습 실패 원인인지 진단용). <0이면 무작위.
flags.DEFINE_integer("fix_env_seed", -1, "If >=0, reset every episode with this seed to pin the sim scene (single-env training/diagnosis). <0 = random each episode.")
flags.DEFINE_float("ep_timeout_secs", 120.0, "Pause update thread if no episode finishes within this many seconds. 0 to disable.")
flags.DEFINE_integer("batch_size", 64, "Mini batch size.")
flags.DEFINE_integer("max_steps", 100_000, "Number of training steps.")
flags.DEFINE_integer("num_data", 0, "Max number of offline demo episodes to load (0 = all).")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean("checkpoint_model", False, "Save agent checkpoint during training.")
flags.DEFINE_integer("checkpoint_interval", 0, "Save agent checkpoint every N steps. When 0 and checkpoint_model=True, no interval saving (save at end only).")
flags.DEFINE_boolean("checkpoint_buffer", False, "Save agent replay buffer on evaluation.")
flags.DEFINE_integer("utd_ratio", 20, "Update to data ratio.")
flags.DEFINE_integer("keep_period", None, "Keep checkpoints every N steps.")
flags.DEFINE_boolean("overwrite", False, "Overwrite existing checkpoint directory.")
flags.DEFINE_boolean("resume", False, "Resume training from checkpoint.")
flags.DEFINE_string("output_dir", "./logs", "Directory for logs and checkpoints.")
flags.DEFINE_integer("fsdp_devices", 1, "Number of FSDP devices for sharding.")

flags.DEFINE_string("client_host", "localhost", "Host for environment operations server.")
flags.DEFINE_integer("client_port", 8102, "Port for environment operations server.")

flags.DEFINE_integer("replan_steps", 8, "Number of replan steps for evaluation.")

# --- 학습 중 주기적 평가(in-training eval) ---
# 학습이 잘 되는지 끝까지 기다리지 않고, 일정 env 스텝마다 에피소드 경계에서 잠시 멈추고
# 결정적 seed로 평가 에피소드를 굴려 성공률/리턴/길이를 찍는다(업데이트/버퍼 삽입 없음).
# async 루프에선 actor(메인 스레드)가 평가 롤아웃을 도는 동안 learner 스레드는 device[1:]
# 에서 계속 업데이트한다(샘플은 device[0]이라 경합 없음). 단, 평가 동안 학습 에피소드가
# 안 끝나 ep_timeout_secs가 지나면 learner가 잠시 멈췄다 다음 실제 에피소드에서 재개한다.
flags.DEFINE_integer("eval_interval", 0, "N env 스텝마다 에피소드 경계에서 평가 실행. 0이면 비활성.")
flags.DEFINE_integer("eval_episodes", 10, "평가 1회당 굴릴 에피소드 수(실시간 페이싱이라 작게).")
flags.DEFINE_integer("eval_seed", -1, "평가 reset seed의 base(sim: seed+ep). <0이면 FLAGS.seed 사용.")
flags.DEFINE_boolean("eval_base_at_start", False, "학습 시작 전 pi0.5 base 정책만 1회 평가(비교 기준선).")

flags.DEFINE_string("dataset_path", "", "Path to the dataset.")
config_flags.DEFINE_config_file(
    "config",
    "configs/model/expo_ft_pi_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

config_flags.DEFINE_config_file(
    "config_task",
    "configs/task/pick.py",
    "File path to the task configuration.",
    lock_config=False,
)

def main(_):
    init_logging()
    assert FLAGS.offline_ratio >= 0.0 and FLAGS.offline_ratio <= 1.0

    jax.config.update(
        "jax_compilation_cache_dir",
        str(epath.Path("~/.cache/jax").expanduser()),
    )

    num_gpus = jax.device_count()
    if num_gpus < 2:
        raise ValueError(
            f"At least 2 GPUs required (1 for sampling, rest for updates), got {num_gpus}"
        )
    sample_device = jax.devices()[0]
    update_devices = jax.devices()[1:]
    num_update = len(update_devices)
    if num_update % FLAGS.fsdp_devices != 0:
        raise ValueError(
            f"Number of update devices ({num_update}) must be divisible by "
            f"fsdp_devices ({FLAGS.fsdp_devices})"
        )
    mesh = jax.sharding.Mesh(
        np.array(update_devices).reshape(num_update // FLAGS.fsdp_devices, FLAGS.fsdp_devices),
        (openpi_sharding.BATCH_AXIS, openpi_sharding.FSDP_AXIS),
    )
    logging.info("Device layout: sampling on %s, updates on %s", sample_device, update_devices)

    data_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(openpi_sharding.DATA_AXIS)
    )
    replicated_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec()
    )
    
    log_dir = os.path.join(FLAGS.output_dir, FLAGS.run_name)
    os.makedirs(log_dir, exist_ok=True)
    train_video_dir = os.path.join(log_dir, "train_videos")
    os.makedirs(train_video_dir, exist_ok=True)
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_dir_path = epath.Path(checkpoint_dir)
    checkpoint_manager, resuming = initialize_checkpoint_dir(
        checkpoint_dir_path,
        keep_period=FLAGS.keep_period,
        overwrite=FLAGS.overwrite,
        resume=FLAGS.resume,
    )

    init_wandb(checkpoint_dir_path, resuming, FLAGS.project_name, FLAGS.run_name)
    wandb.config.update(FLAGS.flag_values_dict(), allow_val_change=resuming)

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
            truncate_at_success=FLAGS.truncate_offline_at_success,
        )
    else:
        raise ValueError(f"Unsupported dataset type: {FLAGS.config_task.env_type}")
    example_action = dataset[0]['actions'][np.newaxis]
    
    # Create training environment wrapper directly
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

    model_cls = FLAGS.config.model_cls
    # BCLearner uses human-intervention chunks for the actor batch only (no critic).
    use_dagger_hil_sampling = model_cls == "BCLearner"
    if model_cls == "BCLearner":
        from expo_ft.agents.alg.bc import load_agent, restore_checkpoint, save_checkpoint
    elif model_cls == "EXPOLearner":
        from expo_ft.agents.alg.expo_ft import load_agent, restore_checkpoint, save_checkpoint
    else:
        raise ValueError(f"Unsupported model class: {model_cls}")

    from expo_ft.agents.vla.pi05 import build_pi05
    actor, actor_train_state, target_actor_params, agent_kwargs, vla_metadata = build_pi05(
        FLAGS.config, FLAGS.seed, mesh, data_sharding, replicated_sharding,
        resuming, env.task_description,
    )

    rb_args = dict(
        config=FLAGS.config,
        example_action=example_action,
        capacity=FLAGS.max_steps,
        task_description=env.task_description,
        replan_steps=FLAGS.replan_steps,
        seed=FLAGS.seed,
    )
    replay_buffer = create_replay_buffer(**rb_args)
    offline_replay_buffer = create_replay_buffer(**rb_args)

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

    agent_example_observation, agent_example_state, agent_example_action = offline_replay_buffer.convert_to_critic_format(
    {
        "base_image": offline_replay_buffer.dataset_dict['base_image'][0][np.newaxis],
        "left_wrist_image": offline_replay_buffer.dataset_dict['left_wrist_image'][0][np.newaxis],
        "state": offline_replay_buffer.dataset_dict['state'][0][np.newaxis],
        "actions": offline_replay_buffer.dataset_dict['actions'][0][np.newaxis],
    })
    actor.action_dim = agent_example_action.squeeze().shape[-1]
    actor.state_dim = agent_example_state.squeeze().shape[-1]
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
    )
    
    start_step = 0
    if resuming:
        agent = restore_checkpoint(checkpoint_manager, agent)
        steps = tuple(checkpoint_manager.all_steps())
        latest_step = max(steps) if steps else None
        if latest_step is not None:
            start_step = latest_step
            logging.info("Resuming from step %d", start_step)
        batch_processor.restore(checkpoint_dir_path, up_to_step=latest_step)

    episode_log = EpisodeState()
    training_log = TrainingStats(
        ep_count=replay_buffer.count_episodes_chronological() if resuming else 0,
    )
    logging.info("Resuming: ep_count set to %d (episodes in replay buffer).", training_log.ep_count)

    batch_processor.on_episode_start()

    dt = 1.0 / FLAGS.config_task.control_hz
    done = False
    # fix_env_seed>=0 → pin sim scene to a single fixed environment every episode
    # (None = random each episode). The per-episode reset below reuses the same value.
    fix_seed = FLAGS.fix_env_seed if FLAGS.fix_env_seed >= 0 else None
    env.reset(seed=fix_seed)
    start_step_time = time.time()
    env.step(FLAGS.config_task.example_action.squeeze().tolist())
    action_plan = deque()
    action_type = "policy"
    combine_rng = jax.random.PRNGKey(FLAGS.seed + 100)

    # --- Async update thread setup ---
    # Actor (main thread) samples on device[0], learner (background thread)
    # updates on device[1:]. Params published via atomic reference swap.
    _actor_agent = agent.cache_infer_params()
    sample_sharding = jax.sharding.SingleDeviceSharding(sample_device)
    _actor_agent.actor.replicated_sharding = sample_sharding
    _published = [None]
    _publish_lock = threading.Lock()
    _buffer_lock = threading.Lock()
    _env_step = [start_step]
    _stop_event = threading.Event()
    _can_update = threading.Event()
    _episode_done = threading.Event()
    # 평가(in-training eval)가 메인 스레드에서 도는 동안 set된다. learner는 이 동안
    # ep_timeout_secs '에피소드 미종료' 타임아웃을 보류한다 — 평가는 학습 에피소드를
    # 진행시키지 않으므로(버퍼 삽입 없음), 길어지면 learner가 헛되이 "pausing updates"를
    # 찍어 wandb update_paused 지표가 1↔0으로 깜빡인다(모니터링 잡음). learner는 평가 동안
    # 계속 업데이트하는 게 정상 동작이므로(샘플 device[0] / 업데이트 device[1:], env 비접근),
    # 타임아웃만 억제하면 된다.
    _eval_active = threading.Event()
    _update_count = [0]
    _ckpt_request = [None]  # main thread sets step number; update thread saves and clears
    _ckpt_done = threading.Event()

    def _update_worker():
        nonlocal combine_rng
        learner_agent = agent
        last_episode_time = time.time()
        update_time = deque(maxlen=10)

        _can_update.wait()
        while not _stop_event.is_set():
            try:
                # 평가 중엔 타임아웃 기준 시각을 계속 끌어올려 헛된 pause 로그/지표 깜빡임을 막는다.
                if _eval_active.is_set():
                    last_episode_time = time.time()
                if FLAGS.ep_timeout_secs > 0 and not _eval_active.is_set() and time.time() - last_episode_time > FLAGS.ep_timeout_secs:
                    logging.info("No episode finished for %.1fs, pausing updates.", FLAGS.ep_timeout_secs)
                    wandb.log({"training/update_paused": 1}, step=_env_step[0])
                    _episode_done.wait()
                    _episode_done.clear()
                    last_episode_time = time.time()
                    logging.info("Episode signal received, resuming updates.")
                    wandb.log({"training/update_paused": 0}, step=_env_step[0])

                if _episode_done.is_set():
                    _episode_done.clear()
                    last_episode_time = time.time()

                with _buffer_lock:
                    batch, actor_batch, combine_rng = batch_processor.next_batch(combine_rng)
                batch_info = get_batch_info(batch)

                t0 = time.time()
                learner_agent = learner_agent.replace(
                    rng=jax.device_put(learner_agent.rng, replicated_sharding)
                )
                learner_agent, update_info = learner_agent.update(
                    learner_agent, batch, FLAGS.utd_ratio, actor_batch
                )

                with _publish_lock:
                    _published[0] = learner_agent._infer_cache

                update_time.append(time.time() - t0)
                _update_count[0] += 1
                log_dict = {f"training/{k}": v for k, v in update_info.items()}
                log_dict["batch_info"] = batch_info
                log_dict["training/num_updates"] = _update_count[0]
                if _update_count[0] % 10 == 0 and len(update_time) == update_time.maxlen:
                    log_dict["training/update_time_avg_ms"] = float(np.mean(update_time)) * 1000.0
                wandb.log(log_dict, step=_env_step[0])

                ckpt_step = _ckpt_request[0]
                if ckpt_step is not None:
                    _ckpt_request[0] = None
                    try:
                        save_checkpoint(checkpoint_manager, learner_agent, ckpt_step)
                        logging.info("Saved agent checkpoint at step %d (from update thread)", ckpt_step)
                    except Exception as e:
                        logging.error("Could not save model checkpoint: %s", e)
                    _ckpt_done.set()
            except Exception:
                logging.exception("Update thread crashed at update %d", _update_count[0])
                break
        # Handle checkpoint request after stop signal
        ckpt_step = _ckpt_request[0]
        if ckpt_step is not None:
            _ckpt_request[0] = None
            try:
                save_checkpoint(checkpoint_manager, learner_agent, ckpt_step)
                logging.info("Saved agent checkpoint at step %d (from update thread)", ckpt_step)
            except Exception as e:
                logging.error("Could not save checkpoint: %s", e)
            _ckpt_done.set()
        logging.info("Update thread exiting (updates=%d).", _update_count[0])

    _update_thread = threading.Thread(target=_update_worker, daemon=True)
    _update_thread.start()

    if resuming and training_log.ep_count >= 10 and replay_buffer._size >= FLAGS.batch_size:
        _can_update.set()
        logging.info("Resuming: replay buffer already warm, update thread starting immediately.")

    # --- 학습 중 평가 헬퍼 ---
    # 메인 스레드(actor)가 평가 롤아웃을 도는 동안 learner 스레드는 그대로 업데이트한다.
    # 평가는 _actor_agent로 샘플하며(파라미터 불변, rng만 진행) 반환 agent를 다시 받는다.
    # 평가 중엔 _published 캐시를 당겨오지 않으므로 한 정책 스냅샷으로 일관되게 측정된다.
    # base_seed: 평가 reset seed의 base. 매 평가 동일 장면 세트 재현(<0이면 FLAGS.seed).
    # active_flag=_eval_active: 평가 롤아웃 동안 learner의 스퍼리어스 타임아웃 pause를
    # 억제한다(평가 중에도 learner는 계속 업데이트).
    evaluator = TrainingEvaluator(
        env=env,
        config_task=FLAGS.config_task,
        num_episodes=FLAGS.eval_episodes,
        base_seed=FLAGS.eval_seed if FLAGS.eval_seed >= 0 else FLAGS.seed,
        fix_env_seed=FLAGS.fix_env_seed,
        replan_steps=FLAGS.replan_steps,
        dt=dt,
        reset_seed=fix_seed,
        active_flag=_eval_active,
    )

    # 학습 시작 전 기준선 평가(step0): pi0.5 base 정책만. learner는 아직 _can_update 대기 중이라
    # 유휴 상태. full(residual+critic)은 step0엔 critic이 랜덤이라 무작위 선택일 뿐이라 재지 않는다.
    if FLAGS.eval_base_at_start:
        _actor_agent = evaluator.run_base(_actor_agent, start_step)
        start_step_time = time.time()
        action_plan.clear()

    last_eval_step = start_step  # 마지막 평가를 돌린 env 스텝(eval_interval 트리거 기준)

    for i in tqdm.tqdm(
        range(start_step, FLAGS.max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm
    ):
        loop_start = time.time()
        step_metrics = {}
        _env_step[0] = i

        with _publish_lock:
            new_cache = _published[0]
            _published[0] = None
        if new_cache is not None:
            _actor_agent = _actor_agent.replace(_infer_cache=new_cache)

        observation = env.get_observation()
        done, success, reward, mask = env.get_info_for_step()

        # Skip model inference while human is controlling.
        if not action_plan and action_type != "human":
            sample_start = time.time()
            action_chunk, _actor_agent, new_si = _actor_agent.sample_actions(observation)
            episode_log.sample_info_history.append(new_si)
            training_log.record_sample_time(time.time() - sample_start, step_metrics)
            action_plan.extend(action_chunk[:FLAGS.replan_steps])
        else:
            episode_log.sample_info_history.append(episode_log.sample_info_history[-1] if episode_log.sample_info_history else None)

        elapsed = time.time() - start_step_time
        if elapsed < dt:
            time.sleep(dt - elapsed)

        has_action = bool(action_plan)
        action = action_plan.popleft() if has_action else np.zeros_like(example_action.squeeze())
        real_action, action_type = env.step(action.tolist())
        start_step_time = time.time()

        episode_log.record_step(observation, len(action_plan), action_type, real_action, reward)

        if action_type == "human":
            action_plan.clear()

        if has_action or action_type == "human":
            transition_dict = dict(
                observations=observation,
                actions=real_action,
                rewards=reward,
                masks=mask,
                dones=done,
                is_hil=(action_type == "human"),
            )
            with _buffer_lock:
                batch_processor.insert_transition(transition_dict)

        if done:
            with _buffer_lock:
                batch_processor.on_episode_done(success)
            _episode_done.set()

            # (eval) 주기적 인-트레이닝 평가: eval_interval env 스텝마다 에피소드 경계에서.
            # learner 스레드는 device[1:]에서 계속 돈다(샘플은 device[0]이라 경합 없음).
            if FLAGS.eval_interval > 0 and (i - last_eval_step) >= FLAGS.eval_interval:
                last_eval_step = i
                _actor_agent = evaluator.run(_actor_agent, i)  # 전체 정책 평가(run_base는 시작 전 기준선용)

            env.reset(seed=fix_seed)

            training_log.on_episode_done(episode_log, success, step_metrics)
            episode_log.reset()
            with _buffer_lock:
                batch_processor.on_episode_start()

            observation = env.get_observation()
            done = False
            action_type = "policy"
            action_plan.clear()

            # don't wait 10 episode to start updating as compiling is slow
            if not _can_update.is_set() and replay_buffer._size >= FLAGS.batch_size:
                _can_update.set()
                logging.info("Replay buffer ready (ep_count=%d), starting update thread.", training_log.ep_count)

        if FLAGS.checkpoint_model and FLAGS.checkpoint_interval > 0 and i > 0 and i % FLAGS.checkpoint_interval == 0:
            _ckpt_done.clear()
            _ckpt_request[0] = i

        if FLAGS.checkpoint_buffer and (has_action or action_type == "human"):
            try:
                save_replay_buffer_transition(checkpoint_dir_path, transition_dict, step=i)
            except Exception:
                logging.exception("Could not save agent buffer.")

        step_metrics["training/loop_time_ms"] = (time.time() - loop_start) * 1000.0
        wandb.log(step_metrics, step=i)

    if FLAGS.checkpoint_model:
        _ckpt_done.clear()
        _ckpt_request[0] = FLAGS.max_steps
    _stop_event.set()
    _can_update.set()
    _episode_done.set()
    _update_thread.join()

    if FLAGS.checkpoint_model:
        logging.info("Waiting for checkpoint manager to finish")
        checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    app.run(main)