"""Batch preparation utilities for critic and actor training."""

import jax.numpy as jnp

# critic 관측에 채널방향으로 쌓을 카메라 키의 기본값.
# DROID는 (3인칭, 왼손목) 2대뿐이고 right_wrist_0_rgb는 zeros/mask=False 더미라
# critic에 넣지 않는 것이 기본이다. robocasa처럼 agentview_right가 실제 영상인
# 태스크는 task config의 critic_camera_keys로 3대를 명시해 넘긴다.
DEFAULT_CRITIC_CAMERA_KEYS = ("base_0_rgb", "left_wrist_0_rgb")


def prepare_critic_batch(batch, padded_dim, action_dim, state_dim, action_horizon, replan_steps,
                         camera_keys=DEFAULT_CRITIC_CAMERA_KEYS):
    """Add critic observations, states, and truncated action targets to a training batch."""
    batch_size = batch["state"].shape[0]

    batch['observations'] = jnp.concatenate(
        [batch["image"][k] for k in camera_keys], axis=-1
    )
    batch['next_observations'] = jnp.concatenate(
        [batch["next_image"][k] for k in camera_keys], axis=-1
    )

    batch['states'] = batch["state"].reshape(batch_size, padded_dim)[..., :state_dim].reshape(batch_size, state_dim)
    batch['next_states'] = batch["next_state"].reshape(batch_size, padded_dim)[..., :state_dim].reshape(batch_size, state_dim)
    batch['critic_states'] = batch['states']
    batch['next_critic_states'] = batch['next_states']

    actions_unpadded = batch["actions"].reshape(batch_size, action_horizon, padded_dim)[..., :action_dim]
    batch['full_actions'] = actions_unpadded.reshape(batch_size, action_horizon * action_dim)
    batch['actions'] = actions_unpadded[:, :replan_steps, :].reshape(batch_size, replan_steps * action_dim)

    return batch


def prepare_actor_sampling_batch(batch):
    """Select next-state fields from a batch for actor action sampling."""
    return {
        "image": batch["next_image"],
        "image_mask": batch["next_image_mask"],
        "state": batch["next_states"],
        "tokenized_prompt": batch['tokenized_prompt'],
        "tokenized_prompt_mask": batch['tokenized_prompt_mask'],
        "token_ar_mask": batch.get('token_ar_mask', None),
        "token_loss_mask": batch.get('token_loss_mask', None),
    }


def extract_critic_fields(processed_inputs, padded_dim, state_dim,
                          camera_keys=DEFAULT_CRITIC_CAMERA_KEYS):
    """Add critic_obs and critic_states keys to a processed inputs dict."""
    processed_inputs["critic_obs"] = jnp.concatenate(
        [processed_inputs["image"][k] for k in camera_keys], axis=-1
    )
    processed_inputs["critic_states"] = processed_inputs["state"].reshape(
        processed_inputs["state"].shape[0], padded_dim
    )[..., :state_dim]
    return processed_inputs
