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


# ── 성공 판정 근거 진단 ──────────────────────────────────────────────────
# robocasa PickPlace 계열 task의 _check_success는 보통 두 조건의 AND다:
#   (1) 대상 객체 obj가 목표 용기 container에 "담김"
#       = 물리 접촉(env.check_contact) AND  obj·container 수평중심거리 < th
#   (2) 그리퍼가 obj에서 충분히 "멀어짐" (gripper_obj_far, 기본 th=0.25m)
# 영상으로는 성공처럼 보이는데 자동판정이 실패로 나오는 대부분의 원인은 (2)다
# (물체는 담겼지만 그리퍼가 25cm 이상 물러나지 않음) 또는 (1)의 수평거리 초과.
# 아래 진단은 매 스텝 두 조건의 수치/충족여부를 계산해, 에피소드 종료 시 사람이
# 읽을 수 있는 근거(어느 조건이 미충족인지)로 로그에 남긴다.
#
# task별 container 객체명·임계값이 다르므로 표로 관리한다. 표에 없는 task는
# container th=None(=robocasa 기본 recep.horizontal_radius*0.7)으로 추정한다.
#   task_name -> (container 객체명, container th[m] 또는 None, gripper-far th[m])
_SUCCESS_SPEC = {
    "PickPlaceCounterToStove": ("container", 0.07, 0.25),
}
_DEFAULT_GRIPPER_FAR_TH = 0.25


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
        # 고정 주방 핀: 둘 다 지정되면 그 (layout, style)로 주방을 고정한다(객체는 무작위 유지).
        fixed_layout_id=None,
        fixed_style_id=None,
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

        # 주방(layout/style)만 단일 장면으로 고정. split="pretrain"은 매 reset마다
        # layout/style를 11~60 풀에서 무작위로 뽑지만(kitchen._setup_model의
        # rng.choice(self.layout_and_style_ids), kitchen.py:595), 후보 리스트를 단일
        # (L,S) 한 쌍으로 줄이면 그 draw가 항상 같은 주방을 반환한다(=주방 고정).
        # 객체 정체성·배치·언어 지시문은 별도 rng draw라 그대로 매 에피소드 무작위로 유지된다
        # (단, 학습 reset에 고정 seed를 주면 안 됨 — 주면 rng가 재시드돼 객체까지 고정됨).
        # 둘 다 지정된 경우에만 적용. 생성자에서만 도는 EXCLUDE 필터는 이 사후 대입을
        # 재검증하지 않으므로 pretrain 풀(11~60) 내 유효 쌍만 쓸 것(예: (11,14)).
        # 생성 시점 RoboCasaGymEnv가 이미 한 번 reset(미고정 풀)했지만, 첫 실제
        # 에피소드의 RoboCasaEnv.reset()이 고정된 리스트로 다시 reset하므로 무해하다.
        if fixed_layout_id is not None and fixed_style_id is not None:
            self.gym.env.layout_and_style_ids = [
                (int(fixed_layout_id), int(fixed_style_id))
            ]
            logger.info(
                "Pinned kitchen to (layout=%d, style=%d); objects/placement/lang remain random",
                int(fixed_layout_id), int(fixed_style_id),
            )

        self._last_obs = None
        self._done = False
        self._success = False
        self._reward = 0.0
        self._steps = 0

        # 에피소드 영상 녹화(3인칭 시점), 에피소드 종료 시 저장.
        self._frame_buffer = []
        self._ep_count = 0

        # 성공 판정 진단용 누적 트래커(reset마다 초기화). _last_diag는 마지막 스텝의
        # 하위조건 측정값, _ever_*는 에피소드 동안 한 번이라도 충족됐는지.
        self._last_diag = None
        self._ever_in_recep = False
        self._ever_gripper_far = False

    # -------------------------------------------------------------------- 헬퍼
    def _diagnose_components(self):
        """현재 sim 상태에서 성공 판정 하위 조건을 계산해 dict로 반환.

        반환 dict 키(측정 실패 시 해당 값은 None):
          obj_in_recep : (충족여부, contact, 수평거리, th, recep명)
          gripper_far  : (충족여부, 그리퍼-obj거리, th)
        _check_success()는 재호출하지 않는다(gym_wrapper가 같은 상태에서 이미
        계산했고 그 결과가 self._success). 어떤 예외도 롤아웃을 깨뜨리지 않도록
        전부 try/except로 감싼다.
        """
        env = self.gym.env
        spec = _SUCCESS_SPEC.get(self.task_name)
        recep_name, recep_th, far_th = (
            spec if spec else ("container", None, _DEFAULT_GRIPPER_FAR_TH)
        )
        comp = {"obj_in_recep": None, "gripper_far": None, "recep": recep_name, "far_th": far_th}

        # (1) obj가 container에 담겼는가: 접촉 AND 수평중심거리 < th
        try:
            from robocasa.utils import object_utils as OU  # noqa: F401 (참고: 기준 구현)

            obj_pos = np.asarray(env.sim.data.body_xpos[env.obj_body_id["obj"]])
            recep_pos = np.asarray(env.sim.data.body_xpos[env.obj_body_id[recep_name]])
            th = (
                recep_th
                if recep_th is not None
                else float(env.objects[recep_name].horizontal_radius * 0.7)
            )
            horiz = float(np.linalg.norm(obj_pos[:2] - recep_pos[:2]))
            contact = bool(env.check_contact(env.objects["obj"], env.objects[recep_name]))
            comp["obj_in_recep"] = (bool(contact and horiz < th), contact, horiz, float(th), recep_name)
        except Exception as e:  # task에 obj/container가 없거나 키가 다를 수 있음
            logger.debug("obj_in_recep 진단 실패: %s", e)

        # (2) 그리퍼가 obj에서 멀어졌는가: 거리 > far_th
        try:
            obj_pos = np.asarray(env.sim.data.body_xpos[env.obj_body_id["obj"]])
            grip_pos = np.asarray(env.sim.data.site_xpos[env.robots[0].eef_site_id["right"]])
            d = float(np.linalg.norm(grip_pos - obj_pos))
            comp["gripper_far"] = (bool(d > far_th), d, float(far_th))
        except Exception as e:
            logger.debug("gripper_far 진단 실패: %s", e)

        return comp

    def _log_episode_outcome(self):
        """에피소드 종료 시 성공/실패 판정 근거를 사람이 읽을 수 있게 로그로 남긴다."""
        comp = self._last_diag or {}
        oir = comp.get("obj_in_recep")
        gf = comp.get("gripper_far")
        recep = comp.get("recep", "container")

        sep = "=" * 72
        result = "SUCCESS" if self._success else "FAILURE"
        lines = [
            "",  # 이전 에피소드 로그와 시각적으로 분리
            sep,
            "[EP %06d] Episode done: %s  steps=%d/%d"
            % (self._ep_count, result, self._steps, self.max_steps),
            sep,
        ]
        # 판정기준(성공 조건) 명시
        far_th = comp.get("far_th", _DEFAULT_GRIPPER_FAR_TH)
        recep_th_txt = f"{oir[3]:.3f}m" if oir else "?"
        lines.append(
            "  판정기준[%s]: obj가 %s에 담김(접촉 AND 수평중심거리<%s) AND 그리퍼-obj거리>%.3fm"
            % (self.task_name, recep, recep_th_txt, far_th)
        )

        blockers = []  # 실패의 직접 원인이 된 미충족 조건들
        # (1) obj-in-receptacle 결과
        if oir is not None:
            ok, contact, horiz, th, rname = oir
            lines.append(
                "  - obj-in-%s : %s (contact=%s, 수평거리=%.3fm %s %.3fm)%s"
                % (
                    rname,
                    "PASS" if ok else "FAIL",
                    contact,
                    horiz,
                    "<" if horiz < th else ">=",
                    th,
                    "" if ok else "  ← 미충족",
                )
            )
            if not ok:
                if not contact:
                    blockers.append(f"{rname} 미접촉(물체가 {rname}에 직접 닿지 않음)")
                elif horiz >= th:
                    blockers.append(f"수평거리 {horiz:.3f}m가 임계 {th:.3f}m 초과")
        # (2) gripper-far 결과
        if gf is not None:
            ok, d, th = gf
            lines.append(
                "  - gripper-far : %s (그리퍼-obj거리=%.3fm %s %.3fm)%s"
                % (
                    "PASS" if ok else "FAIL",
                    d,
                    ">" if d > th else "<=",
                    th,
                    "" if ok else "  ← 미충족",
                )
            )
            if not ok:
                blockers.append(f"그리퍼가 obj에서 {d:.3f}m밖에 안 떨어짐(필요>{th:.3f}m)")

        # 실패라면 원인/힌트 요약
        if not self._success:
            if blockers:
                lines.append("  원인: " + " ; ".join(blockers))
            # "영상상 성공" 전형 패턴: 물체는 담겼는데 그리퍼만 안 물러남
            if oir is not None and oir[0] and gf is not None and not gf[0]:
                lines.append(
                    "  └ 물체는 %s에 담겼으나 그리퍼가 충분히 물러나지 않음 — 영상상 성공처럼 보여도 자동판정은 실패."
                    % recep
                )
            # 에피소드 도중 두 조건이 (다른 시점에라도) 각각 충족된 적이 있는지
            lines.append(
                "  에피소드 중 도달여부: obj-in-%s=%s, gripper-far=%s"
                % (recep, self._ever_in_recep, self._ever_gripper_far)
            )

        lines.append(sep)
        logger.info("\n".join(lines))

    # ------------------------------------------------------------ 추가 내부 헬퍼
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
            logger.info(
                "[EP %06d] Saved episode video: %s (%d frames)",
                self._ep_count, path, len(self._frame_buffer),
            )
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
        # 진단 트래커 초기화
        self._last_diag = None
        self._ever_in_recep = False
        self._ever_gripper_far = False
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

        # 에피소드가 이미 종료된 뒤에도 서버 롤아웃 루프가 reset() 전에 step()을
        # 한 번 더 호출한다(루프가 매 iteration 시작에서 직전 스텝의 done/reward를
        # 읽어 terminal transition을 저장하는 off-by-one 구조 때문). 그 호출까지
        # sim을 진행하면 steps=301/300 같은 가짜 종료 로그와 1프레임짜리 영상이
        # 생긴다. 이미 done이면 sim을 진행하지 않고 마지막 상태를 유지한 채
        # action 형태(arm 7차원)만 돌려준다.
        if self._done:
            return {"executed_action": arm}

        action_dict = convert_action(a12)  # 표준 12차원 -> dict 분리 (env_utils)
        raw, reward, done, _truncated, info = self.gym.step(action_dict)

        self._steps += 1
        self._success = bool(info.get("success", reward > 0))
        self._reward = float(reward)
        # create_env에서 ignore_done=True => done은 success / max_steps로 판정.
        self._done = bool(self._success or done or self._steps >= self.max_steps)

        # 성공 판정 하위조건을 같은 sim 상태에서 측정해 누적(영상상 성공/실제 실패 진단용).
        self._last_diag = self._diagnose_components()
        if self._last_diag.get("obj_in_recep") is not None:
            self._ever_in_recep |= self._last_diag["obj_in_recep"][0]
        if self._last_diag.get("gripper_far") is not None:
            self._ever_gripper_far |= self._last_diag["gripper_far"][0]

        self._last_obs = self._translate_obs(raw)
        self._maybe_record()
        if self._done:
            self._log_episode_outcome()
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
