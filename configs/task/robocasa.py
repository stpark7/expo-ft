"""
RoboCasa365 시뮬레이션 태스크 config: PandaOmron 로봇의 PickPlaceCounterToStove.

이 config를 사용하는 모듈들(둘 다 독립적으로 로드함):

  * client/run_client.py (sim 클라이언트) — env_type / env / env_name / language_instruction를 읽고, 
    dict(config) 전체를 RoboCasaEnv(**kwargs)로 넘긴다. 
    robocasa_* 필드만 RoboCasaEnv가 사용하고 나머지(example_action, control_hz 등)는 **kwargs로 흡수해 무시한다.
  * train_pi_robo.py (server / learner) — env_type / example_action / control_hz / residual_action_xyzg 네 필드만 읽는다.
    롤아웃은 원격 EnvClientWrapper로 실행하므로 서버는 config.env를 사용하지 않는다
    (그래서 robocasa import가 실패하는 server venv에서도 안전).

"""

import ml_collections
import numpy as np

try:
    from client.sim.envs.robocasa_env import RoboCasaEnv
except Exception:
    print("Not importing robocasa env [module]")


def get_config():
    config = ml_collections.ConfigDict()

    # ── env 식별 ─────────────────────────────────────────────────────────
    config.env_type = "sim"
    config.env_name = "robocasa"

    try:
        config.env = RoboCasaEnv
    except Exception:
        print("Not importing robocasa env [env]")

    config.language_instruction = ml_collections.config_dict.placeholder(str)

    # ── RoboCasaEnv 생성 kwargs (client에서 dict(config)로 전달) ──────────
    config.robocasa_task_name = "PickPlaceCounterToStove"

    config.split = "pretrain"

    # ── 고정 주방(kitchen) 핀 ─────────────────────────────────────────────
    # split="pretrain"은 매 reset마다 layout/style를 11~60 풀에서 무작위로 뽑는다.
    # 아래 두 값이 둘 다 지정되면(클라이언트에서 주입) 그 (layout, style) 한 장면으로
    # 주방을 고정한다. 대상 객체 정체성·배치·언어 지시문은 여전히 에피소드마다 무작위
    # (객체/배치는 별도 rng draw라 주방 고정과 독립적). 둘 중 하나라도 None이면 비활성.
    #
    # ⚠️ env는 '클라이언트'만 생성하므로(run_client.py가 이 파일을 직접 다시 읽음),
    #   서버측 --config_task.* 오버라이드로는 이 값이 env에 도달하지 못한다. 실제 활성화는
    #   run_client.py의 --fixed_layout_id/--fixed_style_id (scripts/sim/run_policy.sh)로 한다.
    #   기본 None = 무작위 주방(기존 동작). 값은 pretrain 풀(11~60) 내 유효 쌍만 쓸 것.
    config.fixed_layout_id = ml_collections.config_dict.placeholder(int)
    config.fixed_style_id = ml_collections.config_dict.placeholder(int)

    config.camera_width = 256
    config.camera_height = 256

    # 작업별로 episode 상한이 다르므로, 조정이 필요함
    config.max_steps = 400

    config.example_action = np.zeros((1, 12), dtype=np.float64)
    config.control_hz = 20
    # position + gripper만 수정대상으로 학습
    config.residual_action_xyzg = True

    # critic 관측에 쌓을 카메라들. robocasa는 3인칭 2대(agentview_left/right) +
    # 손목 1대 = 3대가 모두 실제 영상이므로 critic도 9채널(3대)로 본다.
    # (DROID는 right_wrist가 zeros 더미라 기본 2대만 쓴다.)
    config.critic_camera_keys = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")

    # 모델 32차원 action 레이아웃에서 arm(7차원)이 시작하는 위치.
    # ⚠️ robocasa365 체크포인트는 groot 로더(groot_openpi_dataset.py)가 학습 직전
    # action을 ARM-FIRST [eef_pos(0:3), eef_rot(3:6), gripper(6:7), base(7:11),
    # mode(11:12)]로 재배열해 학습했다(=convert_action 순서, norm_stats로 검증됨).
    # 따라서 arm은 모델 [0:7]에 있으므로 offset=0. (LeRobot 데모의 *컬럼* 순서는
    # base-first라 모순처럼 보이지만, 학습 로더가 재배열하므로 체크포인트는 arm-first다.)
    config.action_pad_offset = 0

    return config
