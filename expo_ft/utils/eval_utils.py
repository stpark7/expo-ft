"""학습 중/후 정책 평가(rollout 전용) 공용 유틸.

eval_droid_policy.py(독립 평가 스크립트)와 train_pi_robo*.py(학습 중 주기적 평가)가
'동일한' 롤아웃 로직을 쓰도록 한 곳에 모았다. 두 경로가 갈라지면 학습 중에 보는
성공률과 최종 eval 성공률이 미묘하게 달라져 원인 추적이 불가능해지므로, 단일 소스로 둔다.

평가는 학습과 달리:
  * 그래디언트 업데이트 없음, 리플레이 버퍼 삽입 없음 (env/agent를 읽기만 한다).
  * 에피소드별 '결정적' reset seed로 굴려 실행마다 동일한 장면 세트를 평가한다
    (sim 한정. droid 실로봇은 seed로 장면을 못 고정해 reset_seed=None).
  * sample_actions는 학습과 동일(EXPO: propose→edit→critic select / BC: base만).
    파라미터는 불변이고 agent의 rng만 진행되므로, 호출자는 반환된 agent를 다시 받아야 한다.

env 토폴로지상 평가용 환경이 따로 없다(원격 클라이언트가 환경 1개를 소유). 따라서
학습 루프는 에피소드 경계(done)에서 잠시 멈추고 이 함수로 같은 env에 평가 에피소드를
굴린 뒤, 다시 reset해서 학습을 이어간다. 함수가 끝나면 env는 마지막 평가 에피소드의
종료 상태로 남으므로, 호출자가 학습 재개 전에 env.reset()을 한 번 더 해야 한다.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import jax
import numpy as np


def resolve_max_traj_len(config_task) -> int:
    """에피소드 강제 종료 스텝 상한을 task config에서 도출한다.

    sim(robocasa)은 ignore_done=True라 horizon done을 안 주므로 max_steps로 끊고,
    droid(실로봇)는 auto_reset_steps(자동 리셋까지의 스텝 수)로 끊는다.
    eval_droid_policy.py와 동일한 규칙이어야 학습 중/후 길이 상한이 일치한다.
    """
    if config_task.env_type == "sim":
        return int(config_task.max_steps)
    if config_task.env_type == "droid":
        return int(config_task.auto_reset_steps)
    raise ValueError(f"config_task.env_type must be 'droid' or 'sim', got {config_task.env_type!r}")


def _log_step_timing(logger, ep_index, step, step_t0, timing, done):
    """eval_droid_policy.py와 동일한 per-step 타이밍 로그(제어주기 추종 진단용)."""
    total_ms = (time.time() - step_t0) * 1000.0
    logger.info(
        "[timing][ep %d step %d] total=%.1fms wait=%.1f obs=%.1f info=%.1f plan=%.1f act=%.1f done=%s",
        ep_index + 1, step, total_ms,
        timing["wait_ms"], timing["obs_ms"], timing["info_ms"],
        timing["plan_ms"], timing["act_ms"], done,
    )


def rollout_one_episode(
    env,
    agent,
    *,
    reset_seed: Optional[int],
    replan_steps: int,
    max_traj_len: int,
    dt: float,
    only_base_actions: bool = False,
    log_timing: bool = False,
    logger: Optional[logging.Logger] = None,
    ep_index: int = 0,
) -> Tuple[bool, float, int, str, Any]:
    """평가 에피소드 1회 실행. eval_droid_policy.py의 내부 루프와 byte-동일하게 유지한다.

    Returns:
        (success, ep_return, ep_len, ep_prompt, agent)
        - agent: sample_actions가 rng를 진행시킨 새 agent(파라미터는 불변). 호출자가 다시 받아야 함.
    """
    logger = logger or logging.getLogger(__name__)

    # seed가 주어지면(sim 재현평가) 동일 seed→동일 장면. None이면 무작위(droid/학습 기본).
    observation = env.reset(seed=reset_seed)
    # 이 에피소드에 sim이 샘플한 언어 지시문(=대상 객체). 동일 seed면 실행마다 같아야 한다.
    ep_prompt = observation.get("prompt", "") if isinstance(observation, dict) else ""

    action_plan: deque = deque()  # 샘플한 액션 청크를 한 스텝씩 꺼내 실행하는 큐
    ep_return = 0.0
    ep_len = 0
    success = False
    start_time = time.time()

    # max_traj_len: done(성공)이 안 떠도 에피소드를 강제 종료하는 스텝 상한.
    for step in range(max_traj_len):
        step_t0 = time.time()
        timing = {"wait_ms": 0.0, "obs_ms": 0.0, "info_ms": 0.0, "plan_ms": 0.0, "act_ms": 0.0}

        # (a) 현재 관측 수신
        t_obs0 = time.time()
        observation = env.get_observation()
        timing["obs_ms"] = (time.time() - t_obs0) * 1000.0
        # (b) 직전 스텝의 결과(done/success/reward). 첫 스텝은 reset 직후 상태.
        t_info0 = time.time()
        done, success, reward, _ = env.get_info_for_step()
        timing["info_ms"] = (time.time() - t_info0) * 1000.0

        # (c) 큐가 비었을 때만 정책 추론 → 청크에서 replan_steps개만 큐에 적재.
        #     only_base_actions=True면 residual/critic 없이 pi0.5 base 액션만 사용.
        t_plan0 = time.time()
        if not action_plan:
            action_chunk, agent, _si = agent.sample_actions(
                observation,
                only_base_actions=only_base_actions,
            )
            action_chunk = np.asarray(jax.device_get(action_chunk))  # JAX→numpy(host)
            if action_chunk.ndim == 1:
                action_chunk = action_chunk[None, :]  # 단일 액션이면 [1, dim]으로 보정
            action_plan.extend(list(action_chunk[:replan_steps]))
        timing["plan_ms"] = (time.time() - t_plan0) * 1000.0
        action = action_plan.popleft()  # 이번 스텝에 실행할 액션 하나

        ep_return += reward
        ep_len += 1

        # (d) 직전 스텝에서 done이 떴으면(=성공/종료) 실행하지 않고 에피소드 종료.
        if done:
            if log_timing:
                _log_step_timing(logger, ep_index, step, step_t0, timing, done)
            break

        # (e) 제어 주기(dt) 유지: 직전 스텝 종료 후 dt가 안 지났으면 남는 만큼 대기.
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

        if log_timing:
            _log_step_timing(logger, ep_index, step, step_t0, timing, done)

    return bool(success), float(ep_return), int(ep_len), ep_prompt, agent


def run_eval_episodes(
    env,
    agent,
    *,
    config_task,
    num_episodes: int,
    base_seed: int,
    fix_env_seed: int = -1,
    replan_steps: int = 8,
    only_base_actions: bool = False,
    dt: Optional[float] = None,
    log_timing: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Dict[str, Any], Any]:
    """num_episodes 만큼 평가 에피소드를 굴리고 성공률/리턴/길이를 집계한다.

    reset seed 규칙은 eval_droid_policy.py와 동일:
      * droid(실로봇): seed로 장면 고정 불가 → reset_seed=None.
      * sim + fix_env_seed>=0 : 모든 에피소드를 그 seed '하나'로 단일 고정 장면 평가.
      * sim(기본)            : 에피소드별 seed=base_seed+ep → 실행마다 동일한 장면 세트.

    Returns:
        (result, agent)
        - result: {n, success_rate, mean_return, std_return, mean_len,
                   successes, returns, lengths, prompts}
        - agent: rng가 진행된 새 agent(호출자가 다시 받아야 함).
    """
    logger = logger or logging.getLogger(__name__)
    max_traj_len = resolve_max_traj_len(config_task)
    if dt is None:
        dt = 1.0 / config_task.control_hz

    successes, returns, lengths, prompts = [], [], [], []
    for ep in range(num_episodes):
        if config_task.env_type != "sim":
            reset_seed = None
        elif fix_env_seed >= 0:
            reset_seed = fix_env_seed
        else:
            reset_seed = base_seed + ep

        success, ep_return, ep_len, ep_prompt, agent = rollout_one_episode(
            env, agent,
            reset_seed=reset_seed,
            replan_steps=replan_steps,
            max_traj_len=max_traj_len,
            dt=dt,
            only_base_actions=only_base_actions,
            log_timing=log_timing,
            logger=logger,
            ep_index=ep,
        )
        successes.append(success)
        returns.append(ep_return)
        lengths.append(ep_len)
        prompts.append(ep_prompt)
        logger.info(
            "  [eval ep %d/%d] success=%s return=%.1f len=%d prompt=%r",
            ep + 1, num_episodes, success, ep_return, ep_len, ep_prompt,
        )

    n = len(successes)
    result = {
        "n": n,
        "success_rate": float(np.mean(successes)) if n else 0.0,
        "mean_return": float(np.mean(returns)) if n else 0.0,
        "std_return": float(np.std(returns)) if n else 0.0,
        "mean_len": float(np.mean(lengths)) if n else 0.0,
        "successes": successes,
        "returns": returns,
        "lengths": lengths,
        "prompts": prompts,
    }
    return result, agent


def report_eval(
    step: int,
    result: Dict[str, Any],
    *,
    only_base: bool = False,
    logger: Optional[logging.Logger] = None,
    tag: Optional[str] = None,
) -> Dict[str, Any]:
    """평가 결과를 콘솔에 한 줄로 찍고, wandb에 넣을 dict를 반환한다.

    콘솔 포맷은 사용자가 참고한 distill-flow 로그와 동일하게 맞춘다:
        Eval at step 1000: reward=0.707±0.455, length=202.6, success_rate=0.707 (300 episodes)

    only_base=True면 pi0.5 base 정책만 평가한 기준선이라 'eval_base/*' 네임스페이스로 분리한다.
    """
    logger = logger or logging.getLogger("eval")
    label = " [base]" if only_base else (f" {tag}" if tag else "")
    logger.info(
        "Eval at step %d%s: reward=%.3f±%.3f, length=%.1f, success_rate=%.3f (%d episodes)",
        step, label,
        result["mean_return"], result["std_return"],
        result["mean_len"], result["success_rate"], result["n"],
    )
    prefix = "eval_base" if only_base else "eval"
    return {
        f"{prefix}/success_rate": result["success_rate"],
        f"{prefix}/return": result["mean_return"],
        f"{prefix}/return_std": result["std_return"],
        f"{prefix}/length": result["mean_len"],
        f"{prefix}/episodes": result["n"],
    }


def per_prompt_breakdown(result: Dict[str, Any]) -> Dict[str, Tuple[int, int]]:
    """지시문(대상 객체)별 (성공수, 시도수) 분해. sim에서 객체별 편차를 드러낸다."""
    per_prompt: Dict[str, Tuple[int, int]] = {}
    for p, s in zip(result["prompts"], result["successes"]):
        hit, tot = per_prompt.get(p, (0, 0))
        per_prompt[p] = (hit + int(bool(s)), tot + 1)
    return per_prompt


@dataclass
class TrainingEvaluator:
    """학습 루프의 주기적 인-트레이닝 평가를 한 번 굴리는 헬퍼.

    train_pi_robo.py / train_pi_robo_async.py가 각자 똑같이 들고 있던 `_run_eval`
    클로저를 한 곳으로 모은 것. 고정 설정(env, eval 파라미터, 학습 재개용 reset seed,
    async learner 타임아웃 억제 플래그)을 생성 시 한 번만 받아두고, run()으로 평가
    1회를 돌린다. 호출부는 `agent = evaluator.run(agent, step)` 한 줄이면 된다.

    rollout 자체는 run_eval_episodes(=eval_droid_policy.py와 byte-동일)에 위임하므로,
    학습 중에 보는 성공률과 독립 eval 성공률이 갈라지지 않는다.
    """

    env: Any
    config_task: Any
    num_episodes: int
    base_seed: int             # 평가 reset seed의 base (sim: seed+ep로 동일 장면 세트 재현)
    fix_env_seed: int          # >=0이면 모든 평가 에피소드를 이 seed 하나로 고정
    replan_steps: int
    dt: float
    reset_seed: Optional[int] = None  # 평가 후 학습 재개용 env.reset seed (학습 루프의 fix_seed)
    logger: Optional[logging.Logger] = None
    # async 전용: 평가 롤아웃 동안 set 해두면 learner가 스퍼리어스 타임아웃 pause를 걸지 않음.
    active_flag: Optional[Any] = None

    def __post_init__(self):
        self.logger = self.logger or logging.getLogger("eval")

    def run(self, agent, step: int):
        """전체 정책(pi0.5 base + residual edit + critic select)으로 평가 1회."""
        return self._run(agent, step, only_base=False)

    def run_base(self, agent, step: int):
        """pi0.5 base 정책만으로 평가 1회. RL(residual+critic)이 base를 넘는지 보는 기준선."""
        return self._run(agent, step, only_base=True)

    def _run(self, agent, step: int, *, only_base: bool):
        """평가 에피소드를 굴려 wandb에 기록하고, 학습 재개용으로 env를 다시 reset한다.

        sample_actions는 agent의 rng만 진행시키고 파라미터는 불변이므로(평가),
        rng가 진행된 새 agent를 반환한다. 호출자는 반환값을 다시 받아 이어 써야 한다.
        """
        if self.active_flag is not None:
            self.active_flag.set()
        try:
            result, agent = run_eval_episodes(
                self.env, agent,
                config_task=self.config_task,
                num_episodes=self.num_episodes,
                base_seed=self.base_seed,
                fix_env_seed=self.fix_env_seed,
                replan_steps=self.replan_steps,
                only_base_actions=only_base,
                dt=self.dt,
                logger=self.logger,
            )
        finally:
            if self.active_flag is not None:
                self.active_flag.clear()

        # wandb는 여기서만 쓰므로 지연 import (eval_utils의 rollout 유틸은 wandb-free 유지).
        import wandb
        wandb.log(report_eval(step, result, only_base=only_base, logger=self.logger), step=step)

        # 평가가 env를 평가-에피소드 종료 상태로 남기므로, 학습용으로 다시 reset.
        self.env.reset(seed=self.reset_seed)
        return agent
