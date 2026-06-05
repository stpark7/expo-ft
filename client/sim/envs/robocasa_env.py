"""EXPO-FT 시뮬레이션 클라이언트를 위한 RoboCasa365 시뮬레이션 환경 래퍼.

client/run_client.py가 WebSocket으로 호출하는 gym 형태의
reset / step / get_observation / get_info_for_step 인터페이스를 동일하게
제공한다(실제 로봇 쪽 client/real/envs/droid_env.py를 그대로 미러링).
다만 백엔드는 DROID 하드웨어가 아니라 RoboCasa365 시뮬레이터다.

fork의 표준 RoboCasaGymEnv / PandaOmronKeyConverter를 감싸서
state/action/camera 레이아웃이 robocasa365 pi0.5 체크포인트가 학습된 형태와
정확히 일치하게 만든 뒤, 이를 서버 측 OpenPI RobocasaInputs 변환이 받는
observation/* 규약으로 다시 매핑한다.

핵심 규약 (모두 fork + 체크포인트 norm-stats 기준으로 검증됨):

* 서버에 반환되는 Observation 딕셔너리::

      observation/image        uint8 (H,W,3)  3인칭        (robot0_agentview_left)
      observation/wrist_image  uint8 (H,W,3)  왼쪽 손목     (robot0_eye_in_hand)
      observation/right_image  uint8 (H,W,3)  pi05가 요구함  (robot0_agentview_right)
      observation/state        float32 (16,)  아래 _STATE_ORDER 참고
      prompt                   str            에피소드별 언어 지시문

* observation/state의 concat 순서는 체크포인트를 학습시킨 groot 로더
  (groot_openpi_dataset.py __getitem__)의 ARM-FIRST(eef-first) concat 순서가
  기준(ground truth)이다 (modality.json의 base-first 컬럼순이 아님 — 학습 로더가 재배열함):
  eef_pos_relative(3) · eef_rot_relative(4) · base_position(3) · base_rotation(4)
  · gripper_qpos(2) = 16.  자세한 근거는 아래 _STATE_ORDER 주석 참고.

* Action은 평탄한 12차원 벡터이며, env_utils.convert_action이 이를
  RoboCasaGymEnv.step이 기대하는
  end_effector_position / rotation / gripper_close / base_motion / control_mode
  딕셔너리로 분리한다.

create_env가 ignore_done=True로 동작하므로 robosuite는 horizon에 의한
done을 절대 보고하지 않는다. 따라서 에피소드 종료는 여기서 task 성공 또는
max_steps 상한으로 판정한다.
"""

import logging

import numpy as np

# RoboCasa는 sim 클라이언트 venv에만 존재한다. 여기서 import해도 괜찮다: task
# config가 이 import를 try/except로 감싸므로 서버 venv(robocasa 없음)에서도
# config를 로드할 수 있다. 실제 로봇 쪽 droid env와 동일한 방식이다.
from robocasa.wrappers.gym_wrapper import RoboCasaGymEnv
from robocasa.utils.env_utils import convert_action

logger = logging.getLogger(__name__)

# observation/state 조립 순서.
# ⚠️ 기준은 modality.json의 *컬럼* 순서(base-first)가 아니라, 체크포인트를 학습시킨
# groot 로더(groot_openpi_dataset.py __getitem__)의 concat 순서다. 이 로더는 학습 직전
# state를 ARM-FIRST(eef-first)로 재배열한다:
#   eef_pos_rel(0:3) · eef_rot_rel(3:7) · base_pos(7:10) · base_rot(10:14) · gripper(14:16)
# norm_stats의 지문(std=0 dim이 [10,11]=base_rot, 큰 std가 [7,8]=base_pos)으로 검증됨.
# 예전 base-first 순서는 eef 차원을 base 통계(std=0)로 정규화해 proprio 토큰을 1e6배로
# 폭주시켜 pretrained 정책을 step0부터 망가뜨렸다. 이 순서는 offline 로더와 byte 동일해야 한다.
_STATE_ORDER = (
    "state.end_effector_position_relative", # robot0_base_to_eef_pos   (3)
    "state.end_effector_rotation_relative", # robot0_base_to_eef_quat  (4)
    "state.base_position",                  # robot0_base_pos          (3)
    "state.base_rotation",                  # robot0_base_quat         (4)
    "state.gripper_qpos",                   # robot0_gripper_qpos      (2)
)
_STATE_DIM = 16

# RoboCasaGymEnv 카메라 obs 키 -> 우리 observation/* 규약 키.
_CAMERA_MAP = {
    "observation/image":       "video.robot0_agentview_left",
    "observation/wrist_image": "video.robot0_eye_in_hand",
    "observation/right_image": "video.robot0_agentview_right",
}

ACTION_DIM = 12

# 학습기/리플레이 버퍼는 arm 7차원(eef_pos3 + eef_rot3 + gripper1)만 다룬다
# (residual_action_xyzg). env의 convert_action은 12차원(arm7 + base_motion4 +
# control_mode1)을 기대하므로, 실행 직전 7 -> 12로 re-pad한다. base_motion=0
# (고정베이스), control_mode=-1.0 (LeRobot 데모에서 전 에피소드 일관 실측).
# 학습 action(arm 7)의 (pos,rot,grip) 순서가 convert_action 앞 7자리와 같으므로
# 단순히 뒤에 [0,0,0,0,-1.0]만 붙이면 된다.
_ARM_DIM = 7
_REPAD_TAIL = np.array([0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float64)  # base_motion(4) + control_mode(1)


class RoboCasaEnv:
    """EXPO-FT 클라이언트 인터페이스를 노출하는 gym 형태의 RoboCasa365 환경.

    client/run_client.py가 env(**dict(task_config))에 video_dir
    kwarg를 더해 생성하므로, configs/task/*.py ConfigDict의 모든 필드가
    kwarg로 들어온다. 그중 시뮬레이션과 관련된 것은 소수뿐이고, 나머지
    (DROID bounds, reset_joints, 카메라 serial, spacemouse 속도 등)는
    **kwargs로 무시된다.
    """

    def __init__(
        self,
        video_dir="",
        # robocasa-specific task config fields (set in configs/task/robocasa_*.py)
        robocasa_task_name="PickPlaceCounterToStove",
        split=None,
        camera_width=256,
        camera_height=256,
        max_steps=400,
        language_instruction=None,
        **kwargs,
    ):
        self.video_dir = video_dir or ""
        self.task_name = robocasa_task_name
        self.max_steps = int(max_steps)
        # 고정 fallback 프롬프트; 시뮬레이터의 에피소드별 언어가 이를 덮어쓴다.
        self._default_prompt = language_instruction or ""

        logger.info(
            "Creating RoboCasaEnv(task=%s, split=%s, cam=%dx%d, max_steps=%d)",
            self.task_name, split, camera_width, camera_height, self.max_steps,
        )
        # RoboCasaGymEnv의 기본 split은 "test"인데 create_env가 이를 거부한다;
        # 항상 명시적 split을 넘긴다 (None = 기본 scene).
        self.gym = RoboCasaGymEnv(
            env_name=self.task_name,
            split=split,
            camera_widths=int(camera_width),
            camera_heights=int(camera_height),
            enable_render=True,
        )

        self._last_obs = None
        self._done = False
        self._success = False
        self._reward = 0.0
        self._steps = 0

        # 에피소드 영상 녹화(3인칭 시점), 에피소드 종료 시 저장.
        self._frame_buffer = []
        self._ep_count = 0

    # -------------------------------------------------------------------- 헬퍼
    def _translate_obs(self, raw):
        """RoboCasaGymEnv obs -> 서버가 읽는 observation/* 규약으로 매핑."""
        state = np.concatenate(
            [np.asarray(raw[k], dtype=np.float32).reshape(-1) for k in _STATE_ORDER]
        )
        assert state.shape == (_STATE_DIM,), (
            f"state dim {state.shape} != ({_STATE_DIM},); check proprio keys"
        )
        obs = {"observation/state": state}
        for out_key, cam_key in _CAMERA_MAP.items():
            img = np.asarray(raw[cam_key])
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            obs[out_key] = img
        obs["prompt"] = raw.get("annotation.human.task_description") or self._default_prompt
        return obs

    def _maybe_record(self):
        if self.video_dir and self._last_obs is not None:
            self._frame_buffer.append(self._last_obs["observation/image"])

    def _save_video(self):
        if not (self.video_dir and self._frame_buffer):
            self._frame_buffer = []
            return
        import os
        import imageio
        os.makedirs(self.video_dir, exist_ok=True)
        path = os.path.join(self.video_dir, f"train_ep{self._ep_count:06d}.mp4")
        try:
            imageio.mimwrite(path, self._frame_buffer, fps=20, quality=8)
            logger.info("Saved episode video: %s (%d frames)", path, len(self._frame_buffer))
        except Exception as e:  # 영상 IO가 롤아웃을 깨뜨리지 않게 한다
            logger.warning("Failed to save episode video (%s)", e)
        self._frame_buffer = []
        self._ep_count += 1

    # ------------------------------------------------------------- env 인터페이스
    def reset(self, seed=None):
        # seed가 주어지면 RoboCasaGymEnv가 self.env.rng를 그 시드로 고정한 뒤 reset하므로
        # 레이아웃·스타일·텍스처·대상 객체(=에피소드 언어 지시문)까지 결정적으로 재현된다.
        # eval에서 에피소드별 고정 seed를 주면 '실행마다 동일한 장면 세트'로 평가된다.
        # seed=None이면 매 reset 무작위 샘플(학습 롤아웃 기본 동작).
        raw, _info = self.gym.reset(seed=seed)
        self._steps = 0
        self._done = False
        self._success = False
        self._reward = 0.0
        self._frame_buffer = []
        self._last_obs = self._translate_obs(raw)
        self._maybe_record()
        return self._last_obs

    def step(self, action):
        """학습기의 arm 7차원 action을 실행하고 {"executed_action": arm7}을 반환한다.

        학습기/리플레이 버퍼는 arm 7차원만 쓰지만 env의 convert_action은 12차원을
        기대한다. 따라서 실행 직전 7 -> 12로 re-pad(base_motion=0, control_mode=-1.0)
        하고, '저장용'으로는 다시 7차원 arm을 돌려준다 — 서버 rollout 루프가 step
        반환값(executed_action)을 그대로 버퍼 action으로 저장하므로 action_dim=7과
        일치해야 한다. warmup 등 12차원이 들어오면 앞 12를 그대로 쓰고 arm은 앞 7차원.

        run_client.py는 step_result["executed_action"]만 읽는다
        (reward / done / success는 get_info_for_step으로 따로 가져온다).
        """
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.shape[0] == _ARM_DIM:        # 학습기 arm 7차원 -> 12로 re-pad
            arm = a
            a12 = np.concatenate([arm, _REPAD_TAIL])
        else:                             # warmup 등 12차원: 앞 12 사용, arm은 앞 7
            a12 = a[:ACTION_DIM]
            arm = a12[:_ARM_DIM]
        action_dict = convert_action(a12)  # 표준 12차원 -> dict 분리 (env_utils)
        raw, reward, done, _truncated, info = self.gym.step(action_dict)

        self._steps += 1
        self._success = bool(info.get("success", reward > 0))
        self._reward = float(reward)
        # create_env에서 ignore_done=True => done은 success / max_steps로 판정.
        self._done = bool(self._success or done or self._steps >= self.max_steps)

        self._last_obs = self._translate_obs(raw)
        self._maybe_record()
        if self._done:
            logger.info(
                "Episode done: success=%s steps=%d/%d",
                self._success, self._steps, self.max_steps,
            )
            self._save_video()
        return {"executed_action": arm}

    def get_observation(self):
        # step/reset이 변환된 observation을 이미 캐시해 두었다.
        return self._last_obs

    def get_info_for_step(self):
        """(done, success, reward, mask) — mask는 n-step 연속 플래그."""
        mask = 0.0 if self._done else 1.0
        return self._done, self._success, self._reward, mask

    def close(self):
        try:
            self.gym.close()
        except Exception:
            pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


__all__ = ["RoboCasaEnv"]
