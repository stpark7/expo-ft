"""Image augmentation utilities for Pi training."""

from typing import Callable, Dict

import augmax
import jax
import jax.numpy as jnp

# 증강 대상 카메라 키의 표준 순서. image_dict에 존재하는 것만 증강한다
# (DROID는 right_wrist_0_rgb가 zeros 더미라 사실상 무의미하지만 증강해도 무해;
#  robocasa는 agentview_right가 실제 영상이므로 다른 카메라와 동일하게 증강된다).
CAMERA_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def _crop_resize_rotate_jitter(width, height):
    return [
        augmax.RandomCrop(int(width * 0.95), int(height * 0.95)),
        augmax.Resize(width, height),
        augmax.Rotate((-5, 5)),
        augmax.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    ]


def batched_openpi_augmentation(rng, image_dict: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
    """Apply same image augmentation as openpi preprocess_observation.

    image_dict는 [-1, 1] float32 카메라 텐서들을 담는다(CAMERA_KEYS 중 존재하는 것).
    각 카메라마다 독립 rng로 RandomCrop(95%), Resize, Rotate(-5,5), ColorJitter 적용.
    Returns augmented image dict.
    """
    keys = [k for k in CAMERA_KEYS if k in image_dict]
    sample = image_dict[keys[0]]
    height, width = sample.shape[1], sample.shape[2]
    batch_size = sample.shape[0]

    # 카메라별로 독립적인 rng를 뽑아 서로 다른 crop/rotate/jitter가 걸리게 한다.
    sub_rngs = jax.random.split(rng, len(keys) * batch_size)

    augmented = {}
    for i, k in enumerate(keys):
        img = image_dict[k] / 2.0 + 0.5
        rngs = sub_rngs[i::len(keys)]
        img = jax.vmap(augmax.Chain(*_crop_resize_rotate_jitter(width, height)))(rngs, img)
        augmented[k] = img * 2.0 - 1.0

    return dict(image_dict, **augmented)


def random_crop(key, img: jnp.ndarray, padding: int) -> jnp.ndarray:
    crop_from = jax.random.randint(key, (2,), 0, 2 * padding + 1)
    crop_from = jnp.concatenate([crop_from, jnp.zeros((1,), dtype=jnp.int32)])
    padded_img = jnp.pad(
        img, ((padding, padding), (padding, padding), (0, 0)), mode="edge"
    )
    return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)


def batched_random_crop_per_image(key, obs: jnp.ndarray, padding: int = 4) -> jnp.ndarray:
    """Apply random crop independently to each 3-channel camera stacked in obs.

    obs is (B, H, W, 3 * n_cam): 채널방향으로 카메라가 3채널씩 쌓여 있다.
    각 카메라가 독립적인 crop offset을 받는다.
    """
    n_cam = obs.shape[-1] // 3
    keys = jax.random.split(key, n_cam * obs.shape[0])
    crops = []
    for i in range(n_cam):
        cam = obs[..., i * 3:(i + 1) * 3]
        cam = jax.vmap(random_crop, (0, 0, None))(keys[i::n_cam], cam, padding)
        crops.append(cam)
    return jnp.concatenate(crops, axis=-1)


def batched_crop_only_augmentation(rng, image_dict: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
    """Apply only random crop (padding=12) independently to each present camera."""
    keys = [k for k in CAMERA_KEYS if k in image_dict]
    obs = jnp.concatenate([image_dict[k] for k in keys], axis=-1)
    obs = batched_random_crop_per_image(rng, obs, padding=12)
    augmented = {k: obs[..., i * 3:(i + 1) * 3] for i, k in enumerate(keys)}
    return dict(image_dict, **augmented)


def make_data_augmentation_fn(
    use_full_augmentation: bool = True,
) -> Callable[[jax.Array, Dict[str, jnp.ndarray]], Dict[str, jnp.ndarray]]:
    """Return rng-keyed augmentation fn for learner create()."""

    def data_augmentation_fn(rng, image_dict):
        if use_full_augmentation:
            return batched_openpi_augmentation(rng, image_dict)
        return batched_crop_only_augmentation(rng, image_dict)

    return data_augmentation_fn
