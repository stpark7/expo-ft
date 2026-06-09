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

# LeRobot parquet의 *컬럼* 레이아웃은 base-first(meta/modality.json 기준):
#   action : base_motion[0:4] · control_mode[4:5] · eef_pos[5:8] · eef_rot[8:11] · gripper[11:12]
#   state  : base_pos[0:3] · base_rot[3:7] · eef_pos_rel[7:10] · eef_rot_rel[10:14] · gripper[14:16]
# ⚠️ 그러나 체크포인트는 groot 로더가 학습 직전 ARM-FIRST(eef-first)로 재배열한 형태로
# 학습됐다(norm_stats로 검증). 따라서 여기서 action은 arm 구간 [5:12]만 떼면
# 그 자체가 arm-first 내부순서 [eef_pos3, eef_rot3, gripper1]가 되어 OK지만,
# state(16차원 전체)는 base-first→arm-first로 재배열해야 한다(아래 process_robocasa_dataset).
# 학습기 7차원 = arm = parquet action[5:12] (eef_pos3 + eef_rot3 + gripper1).
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
    num_data=None,
    truncate_at_success=False,
    reward_success_threshold=0.5):
    """
    오프라인 RoboCasa365 LeRobot 데모를 리플레이 버퍼 transition 리스트로 펼친다.

    핵심 변환:
      * action : LeRobot 12차원에서 arm 7차원(action[5:12]=eef_pos·eef_rot·gripper)만
                 추출 -> 학습기 action_dim=7. base_motion·control_mode는 드롭.
      * state  : observation.state(16,)를 base-first→arm-first(eef-first)로 재배열
                 (env _STATE_ORDER와 동일; 체크포인트 학습 순서에 맞춤).
      * images : 3개 mp4 디코딩 -> observation/image(agentview_left)·
                 wrist_image(eye_in_hand)·right_image(agentview_right), uint8 (H,W,3).
      * reward/done : parquet next.reward / next.done 그대로(온라인 env reward와 일관).
      * prompt : task_index -> meta/tasks.jsonl 문자열.

    n-step value 일관성(truncate_at_success):
        온라인 env(robocasa_env)는 첫 성공에서 즉시 종료(done/mask=0)하므로 성공
        transition의 n-step critic 타깃이 ≈1.0이다. 반면 LeRobot 데모는 성공 보상
        (reward=1)을 마지막 ~16프레임 동안 그대로 유지(held)하고 done은 맨 끝 1프레임에만
        둔다. 그러면 첫 성공 프레임의 n-step 타깃이 Σγ^i(=replan_steps=8·γ=0.99이면
        7.726)+부트스트랩으로 부풀어, 같은 성공 사건에 대해 offline(≈7.7) / online(≈1.0)
        critic 타깃이 어긋난다(DICE-RL ph_pretrain↔ph_finetune 불일치).
        truncate_at_success=True면 각 데모를 '첫 성공 프레임'에서 잘라 그 프레임을
        done=1/mask=0으로 만든다 → 온라인과 동일한 단일-terminal 보상(타깃 ≈1.0).
        trailing held-reward 프레임은 버린다(거의 중복 success state라 정보 손실 미미).
        디스크 데이터는 건드리지 않고 로드 시에만 적용하므로 플래그로 A/B 가능.

    Args:
        datapath: LeRobot 데이터셋 루트(meta/info.json 보유). 'lerobot/' 하위로 한
            단계 들어가야 하면 자동 처리.
        task_config: 사용하지 않음(process_droid_dataset과 시그니처를 맞추기 위함).
        episode_indices: 사용할 에피소드 인덱스 목록. None이면 전체(또는 num_data).
        num_data: episode_indices가 None일 때 앞에서부터 사용할 에피소드 수.
        truncate_at_success: True면 첫 성공 프레임에서 에피소드를 자르고 done=1로 표시.
        reward_success_threshold: 성공 판정 reward 임계값(sparse 0/1이므로 0.5).

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
    _trunc_demos = 0       # 첫 성공에서 잘린 데모 수
    _trunc_frames = 0      # 버려진 trailing 프레임 총합
    _no_success_demos = 0  # 성공 프레임이 없어 자르지 못한 데모 수
    for ep in tqdm(ep_list):
        chunk = ep // chunks_size
        pq = os.path.join(root, "data", f"chunk-{chunk:03d}", f"episode_{ep:06d}.parquet")
        df = pd.read_parquet(pq)
        T = len(df)

        actions = np.stack(df["action"].to_numpy())[:, _ROBOCASA_ARM_SLICE].astype(np.float32)  # (T,7) arm-first
        # parquet observation.state는 base-first. 체크포인트가 기대하는 arm-first(eef-first)로
        # 재배열한다(robocasa_env._STATE_ORDER와 byte 동일해야 함).
        states_raw = np.stack(df["observation.state"].to_numpy()).astype(np.float32)             # (T,16) base-first
        states = np.concatenate(
            [states_raw[:, 7:10],    # eef_pos_rel
             states_raw[:, 10:14],   # eef_rot_rel
             states_raw[:, 0:3],     # base_pos
             states_raw[:, 3:7],     # base_rot
             states_raw[:, 14:16]],  # gripper_qpos
            axis=1,
        )                                                                                        # (T,16) arm-first
        rewards = df["next.reward"].to_numpy().astype(np.float32)                                 # (T,)
        dones = df["next.done"].to_numpy().astype(np.float32)                                     # (T,)
        masks = (1.0 - dones).astype(np.float32)                                                  # (T,)
        prompt = tasks.get(int(df["task_index"].iloc[0]), "")

        # 첫 성공에서 truncate: 온라인(첫 성공 종료)과 n-step 타깃 일관성 맞춤.
        T_eff = T
        if truncate_at_success:
            success_mask = rewards > reward_success_threshold
            if success_mask.any():
                success_idx = int(np.argmax(success_mask))   # 첫 성공 프레임
                T_eff = success_idx + 1                       # 그 프레임까지만 사용
                dones = dones.copy(); dones[success_idx] = 1.0
                masks = masks.copy(); masks[success_idx] = 0.0
                _trunc_demos += 1
                _trunc_frames += (T - T_eff)
            else:
                _no_success_demos += 1                        # 성공 없음 → 전체 유지

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

        for t in range(T_eff):
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

    if truncate_at_success:
        avg_trim = (_trunc_frames / _trunc_demos) if _trunc_demos else 0.0
        print(
            f"[truncate_at_success] 잘린 데모 {_trunc_demos}/{len(ep_list)}개, "
            f"버린 프레임 {_trunc_frames}개(데모당 평균 {avg_trim:.1f}프레임), "
            f"성공없음 {_no_success_demos}개 → 단일-terminal 보상(타깃≈1.0)으로 정렬."
        )

    return data
