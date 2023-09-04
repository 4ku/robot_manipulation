from ml_collections import ConfigDict

from dataclasses import dataclass
from typing import Optional


@dataclass
class SAC_RND_Config:
    # wandb params
    project: str = "SAC-RND"
    group: str = "sac-rnd"
    name: str = "sac-rnd"
    # model params
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    alpha_learning_rate: float = 3e-4
    hidden_dim: int = 128
    gamma: float = 0.99
    tau: float = 5e-3
    beta: float = 1.0
    num_critics: int = 2
    critic_layernorm: bool = True
    # rnd params
    rnd_learning_rate: float = 3e-4
    rnd_hidden_dim: int = 128
    rnd_embedding_dim: int = 16
    rnd_mlp_type: str = "concat_first"
    rnd_target_mlp_type: Optional[str] = None
    rnd_switch_features: bool = True
    rnd_update_epochs: int = 50
    # training params
    num_epochs: int = 50
    num_updates_on_epoch: int = 100
    normalize_reward: bool = False
    # general params
    train_seed: int = 10


# Goal-conditioned BC
def get_config(config_string):
    base_real_config = dict(
        batch_size=32,
        eval_interval=1, # num epochs
        save_interval=1, # num epochs
        save_dir="results",
        data_path="/home/ivan/Desktop/bridge_data_v2/data/bridge_tfrecord",
        resume_path=None,
        seed=42,
    )

    base_data_config = dict(
        shuffle_buffer_size=25000,
        prefetch_num_batches=20,
        augment=True,
        augment_next_obs_goal_differently=False,
        augment_kwargs=dict(
            random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
            random_brightness=[0.2],
            random_contrast=[0.8, 1.2],
            random_saturation=[0.8, 1.2],
            random_hue=[0.1],
            augment_order=[
                "random_resized_crop",
                "random_brightness",
                "random_contrast",
                "random_saturation",
                "random_hue",
            ],
        ),
    )

    possible_structures = {
        "gc_bc": ConfigDict(
            dict(
                agent="gc_bc",
                agent_kwargs=dict(
                    network_kwargs=dict(
                        hidden_dims=(256, 256, 256),
                        dropout_rate=0.1,
                    ),
                    policy_kwargs=dict(
                        tanh_squash_distribution=False,
                        fixed_std=[1, 1, 1, 1, 1, 1, 1],
                        state_dependent_std=False,
                    ),
                    early_goal_concat=True,
                    shared_goal_encoder=True,
                    use_proprio=False,
                    learning_rate=3e-4,
                    warmup_steps=2000,
                    decay_steps=int(2e6),
                ),
                dataset_kwargs=dict(
                    goal_relabeling_strategy="uniform",
                    goal_relabeling_kwargs=dict(reached_proportion=0.0),
                    relabel_actions=True,
                    **base_data_config,
                ),
                encoder="resnetv1-18-bridge",
                encoder_kwargs=dict(
                    pooling_method="avg",
                    add_spatial_coordinates=True,
                    act="swish",
                ),
                **base_real_config,
            )
        ),
    }

    return possible_structures[config_string]
