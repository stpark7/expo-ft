"""
RoboCasa365 (시뮬레이션) LeRobot 데모 로더.

LeRobot v2.1 포맷(parquet = state/action/reward/done, mp4 = 카메라 3개)을 읽어, 
서버의 리플레이 버퍼에 넣을 타임스텝 transition 리스트로 펼친다. 
출력 transition은 client/sim/envs/robocasa_env.RoboCasaEnv가 온라인에서 내보내는 observation/* 키 구조와 '동일'하다
(서버의 동일한 OpenPI transform 파이프라인을 그대로 타도록 하기 위함).

robocasa(클라이언트 venv 전용) 자체는 import하지 않으므로 서버 venv에서 안전하다.
parquet/비디오 디코딩은 서버 venv에 있는 pandas / imageio로 처리한다.
"""

import os

import numpy as np
from tqdm import tqdm

# LeRobot 원본 action(12) 레이아웃(meta/modality.json 기준):
#   base_motion[0:4] · control_mode[4:5] · eef_pos[5:8] · eef_rot[8:11] · gripper[11:12]
# 학습기 7차원 = arm = action[5:12] (eef_pos3 + eef_rot3 + gripper1).
_ROBOCASA_ARM_SLICE = slice(5, 12)

# robocasa_env._CAMERA_MAP와 동일(LeRobot 카메라 키 -> observation/* 규약).
_ROBOCASA_VIDEO_MAP = {
    "observation/image":       "observation.images.robot0_agentview_left",
    "observation/wrist_image": "observation.images.robot0_eye_in_hand",
    "observation/right_image": "observation.images.robot0_agentview_right",
}


def _resolve_lerobot_root(datapath):
    """datapath가 LeRobot 루트(meta/info.json 보유)인지 확인해 경로를 돌려준다.
    'lerobot/' 하위로 한 단계 더 들어가야 하는 경우도 자동 처리한다."""
    if os.path.exists(os.path.join(datapath, "meta", "info.json")):
        return datapath
    cand = os.path.join(datapath, "lerobot")
    if os.path.exists(os.path.join(cand, "meta", "info.json")):
        return cand
    raise FileNotFoundError(
        f"LeRobot meta/info.json를 {datapath!r} (또는 그 lerobot/ 하위)에서 찾지 못했다. "
        "datapath는 LeRobot 데이터셋 루트를 가리켜야 한다."
    )


def process_robocasa_dataset(
    datapath,
    task_config=None,
    episode_indices=None,
    num_data=None):
    """
    오프라인 RoboCasa365 LeRobot 데모를 리플레이 버퍼 transition 리스트로 펼친다.

    핵심 변환:
      * action : LeRobot 12차원에서 arm 7차원(action[5:12]=eef_pos·eef_rot·gripper)만
                 추출 -> 학습기 action_dim=7. base_motion·control_mode는 드롭.
      * state  : observation.state(16,) 그대로(modality.json 순서 = env _STATE_ORDER).
      * images : 3개 mp4 디코딩 -> observation/image(agentview_left)·
                 wrist_image(eye_in_hand)·right_image(agentview_right), uint8 (H,W,3).
      * reward/done : parquet next.reward / next.done 그대로(온라인 env reward와 일관).
      * prompt : task_index -> meta/tasks.jsonl 문자열.

    Args:
        datapath: LeRobot 데이터셋 루트(meta/info.json 보유). 'lerobot/' 하위로 한
            단계 들어가야 하면 자동 처리.
        task_config: 사용하지 않음(process_droid_dataset과 시그니처를 맞추기 위함).
        episode_indices: 사용할 에피소드 인덱스 목록. None이면 전체(또는 num_data).
        num_data: episode_indices가 None일 때 앞에서부터 사용할 에피소드 수.

    Returns:
        타임스텝 transition dict 리스트. 각 dict는 observations, actions(7,), rewards,
        masks(=1-dones), dones 키를 가진다(process_droid_dataset과 동형).

    Note:
        이미지를 메모리에 펼치므로(에피소드당 약 150MB) 대량 로드 시 num_data로
        에피소드 수를 제한하라.
    """
    import json as _json

    import imageio.v3 as iio
    import pandas as pd

    root = _resolve_lerobot_root(datapath)
    with open(os.path.join(root, "meta", "info.json")) as f:
        info = _json.load(f)
    chunks_size = int(info.get("chunks_size", 1000))
    total_eps = int(info["total_episodes"])

    # task_index -> 언어 프롬프트
    tasks = {}
    with open(os.path.join(root, "meta", "tasks.jsonl")) as f:
        for line in f:
            rec = _json.loads(line)
            tasks[int(rec["task_index"])] = rec["task"]

    if episode_indices is not None:
        ep_list = [i for i in episode_indices if 0 <= i < total_eps]
    elif num_data is not None and num_data > 0:
        ep_list = list(range(total_eps))[:num_data]
    else:
        ep_list = list(range(total_eps))

    print(f"Find {total_eps} episodes; using {len(ep_list)}")

    data = []
    for ep in tqdm(ep_list):
        chunk = ep // chunks_size
        pq = os.path.join(root, "data", f"chunk-{chunk:03d}", f"episode_{ep:06d}.parquet")
        df = pd.read_parquet(pq)
        T = len(df)

        actions = np.stack(df["action"].to_numpy())[:, _ROBOCASA_ARM_SLICE].astype(np.float32)  # (T,7)
        states = np.stack(df["observation.state"].to_numpy()).astype(np.float32)                 # (T,16)
        rewards = df["next.reward"].to_numpy().astype(np.float32)                                 # (T,)
        dones = df["next.done"].to_numpy().astype(np.float32)                                     # (T,)
        masks = (1.0 - dones).astype(np.float32)                                                  # (T,)
        prompt = tasks.get(int(df["task_index"].iloc[0]), "")

        # 카메라 3개 디코딩 (각 (T,H,W,3) uint8). 프레임 수는 parquet 행과 1:1.
        # plugin="pyav": in-process 디코딩(av). 기본 ffmpeg 플러그인은 subprocess를
        # fork하는데, JAX(멀티스레드) 초기화 후 fork는 데드락 위험이 있어 피한다.
        imgs = {}
        for out_key, cam_key in _ROBOCASA_VIDEO_MAP.items():
            vp = os.path.join(root, "videos", f"chunk-{chunk:03d}", cam_key, f"episode_{ep:06d}.mp4")
            frames = np.asarray(iio.imread(vp, plugin="pyav"))
            if frames.dtype != np.uint8:
                frames = np.clip(frames, 0, 255).astype(np.uint8)
            assert len(frames) == T, (
                f"ep{ep} {cam_key}: 비디오 프레임 {len(frames)} != parquet 행 {T}"
            )
            imgs[out_key] = frames

        for t in range(T):
            observations = {"observation/state": states[t], "prompt": prompt}
            for out_key in _ROBOCASA_VIDEO_MAP:
                observations[out_key] = imgs[out_key][t]
            data.append({
                "observations": observations,
                "actions": actions[t],
                "rewards": rewards[t],
                "masks": masks[t],
                "dones": dones[t],
            })

    return data
