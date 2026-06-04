import os

import h5py
import numpy as np
from tqdm import tqdm

def _discover_episode_dirs(base_path):
    dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    # Keep only numeric directory names (episode indices); skip e.g. action_videos, lerobot
    dirs = [d for d in dirs if d.isdigit()]
    dirs = sorted(dirs, key=lambda x: int(x))
    return [os.path.join(base_path, d) for d in dirs]

def process_droid_dataset(
    datapath,
    task_config,
    episode_indices = None,
    num_data = None):
    """오프라인 DROID 데모(HDF5)를 리플레이 버퍼에 넣을 transition 리스트로 펼친다.

    `datapath` 아래의 각 에피소드 디렉터리에서 `traj.hdf5`를 읽어, 한 궤적을
    타임스텝 단위 transition 딕셔너리들로 분해한다. 팔 액션과 그리퍼 액션은
    하나의 액션 벡터로 합치고, 데모는 성공 궤적으로 간주해 마지막 스텝에만
    reward=1, done=1을 부여한다.

    Args:
        datapath: 에피소드 디렉터리들을 담은 루트 경로. 숫자 이름 디렉터리만
            에피소드로 인식한다(action_videos, lerobot 등은 무시).
        task_config: 액션 키 설정. `action_space`(팔)와
            `gripper_action_space`(그리퍼)를 이어붙여 액션 벡터를 만든다.
        episode_indices: 사용할 에피소드 인덱스 목록. None이면 전체(또는 num_data).
        num_data: episode_indices가 None일 때 앞에서부터 사용할 에피소드 수.

    Returns:
        타임스텝 transition dict의 리스트. 각 dict는 observations, actions,
        rewards, masks(=1-dones), dones 키를 가진다.
    """
    ep_dirs = _discover_episode_dirs(datapath)
    if episode_indices is not None:
        ep_dirs = [ep_dirs[i] for i in episode_indices if 0 <= i < len(ep_dirs)]
    elif num_data is not None and num_data > 0:
        ep_dirs = ep_dirs[:num_data]
    
    print(f"Find {len(_discover_episode_dirs(datapath))} episodes; using {len(ep_dirs)}")

    data = []
    for ep in tqdm(ep_dirs):
        with h5py.File(os.path.join(ep, "traj.hdf5"), "r") as f:
            def load_recursive(group):
                result = {}
                for k, v in group.items():
                    if isinstance(v, h5py.Group):
                        result[k] = load_recursive(v)
                    else:
                        arr = np.asarray(v)
                        # Decode bytes to strings (h5py stores strings as bytes)
                        if arr.dtype.kind == 'S':
                            result[k] = arr.astype('U')
                        elif arr.dtype == object and arr.size > 0 and isinstance(arr.flat[0], bytes):
                            result[k] = np.array([s.decode('utf-8') for s in arr.flat]).reshape(arr.shape)
                        else:
                            result[k] = arr
                return result
            
            ep_obs = load_recursive(f["saved_observation"])
            
            action_key = task_config.action_space
            gripper_key = f"gripper_{task_config.gripper_action_space}"
            a1 = np.asarray(f["action"][action_key])
            a2 = np.asarray(f["action"][gripper_key])
            ep_actions = np.concatenate([a1, a2[:, None] if len(a2.shape) == 1 else a2], axis=-1)
            
            T = len(ep_actions)
            ep_dones = np.pad(np.array([1.0], dtype=np.float32), (T-1, 0), constant_values=0)
            ep_rewards = np.pad(np.array([1.0], dtype=np.float32), (T-1, 0), constant_values=0)
            
            def extract_t(obs, t):
                return {k: extract_t(v, t) if isinstance(v, dict) else (v[t] if isinstance(v, np.ndarray) and len(v.shape) > 0 else v)
                       for k, v in obs.items()}
            
            for t in range(T):
                data.append({
                    "observations": extract_t(ep_obs, t),
                    "actions": ep_actions[t],
                    "rewards": ep_rewards[t],
                    "masks": 1 - ep_dones[t],
                    "dones": ep_dones[t]
                })

    return data
