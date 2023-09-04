import gc
import tensorflow as tf
import numpy as np
import jax
import jax.numpy as jnp
from absl import app, flags, logging
from ml_collections import config_flags
from flax.training import checkpoints
import os
import optax
from functools import partial
from tqdm.auto import trange
from flax.training.train_state import TrainState

from jaxrl_m.common.wandb import WandBLogger
from jaxrl_m.common.common import shard_batch
from jaxrl_m.data.bridge_dataset import BridgeDataset, glob_to_path_list
from jaxrl_m.vision import encoders
from jaxrl_m.agents import agents

from sac_rnd.offline_sac.networks import RND, EnsembleCritic, Alpha
from sac_rnd.offline_sac.utils.common import Metrics
from sac_rnd.offline_sac.utils.running_moments import RunningMeanStd

from config import SAC_RND_Config
from sac_rnd_custom import (
    RNDTrainState,
    CriticTrainState,
    get_flat_obs,
    update_rnd,
    update_sac,
)

try:
    from jax_smi import initialise_tracking  # type: ignore

    initialise_tracking()
except ImportError:
    pass

FLAGS = flags.FLAGS

flags.DEFINE_string("name", "", "Experiment name.")
flags.DEFINE_bool("debug", False, "Debug config")

config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

config_flags.DEFINE_config_file(
    "bridgedata_config",
    None,
    "File path to the bridgedata configuration.",
    lock_config=False,
)


def load_datasets():
    assert type(FLAGS.bridgedata_config.include[0]) == list
    task_paths = [
        glob_to_path_list(
            path, prefix=FLAGS.config.data_path, exclude=FLAGS.bridgedata_config.exclude
        )
        for path in FLAGS.bridgedata_config.include
    ]

    train_paths = [
        [os.path.join(path, "train/out.tfrecord") for path in sub_list]
        for sub_list in task_paths
    ]
    val_paths = [
        [os.path.join(path, "val/out.tfrecord") for path in sub_list]
        for sub_list in task_paths
    ]

    obs_horizon = FLAGS.config.get("obs_horizon")

    train_data = BridgeDataset(
        train_paths,
        FLAGS.config.seed,
        batch_size=FLAGS.config.batch_size,
        train=True,
        action_metadata=FLAGS.bridgedata_config.action_metadata,
        sample_weights=FLAGS.bridgedata_config.sample_weights,
        obs_horizon=obs_horizon,
        **FLAGS.config.dataset_kwargs,
    )
    val_data = BridgeDataset(
        val_paths,
        FLAGS.config.seed,
        batch_size=FLAGS.config.batch_size,
        action_metadata=FLAGS.bridgedata_config.action_metadata,
        train=False,
        obs_horizon=obs_horizon,
        **FLAGS.config.dataset_kwargs,
    )
    return train_data, val_data


def create_actor_agent(example_batch):
    # define encoder
    encoder_def = encoders[FLAGS.config.encoder](**FLAGS.config.encoder_kwargs)

    # initialize agent
    rng = jax.random.PRNGKey(FLAGS.config.seed)
    rng, construct_rng = jax.random.split(rng)
    agent = agents[FLAGS.config.agent].create(
        rng=construct_rng,
        observations=example_batch["observations"],
        goals=example_batch["goals"],
        actions=example_batch["actions"],
        encoder_def=encoder_def,
        **FLAGS.config.agent_kwargs,
    )
    if FLAGS.config.resume_path is not None:
        agent = checkpoints.restore_checkpoint(FLAGS.config.resume_path, target=agent)

    return agent


def main(_):
    devices = jax.local_devices()
    num_devices = len(devices)
    assert FLAGS.config.batch_size % num_devices == 0

    # prevent tensorflow from using GPUs
    tf.config.set_visible_devices([], "GPU")

    # set up wandb and logging
    wandb_config = WandBLogger.get_default_config()
    wandb_config.update(
        {
            "project": "jaxrl_m_bridgedata",
            "exp_descriptor": FLAGS.name,
        }
    )
    wandb_logger = WandBLogger(
        wandb_config=wandb_config,
        variant=FLAGS.config.to_dict(),
        debug=FLAGS.debug,
    )

    save_dir = tf.io.gfile.join(
        FLAGS.config.save_dir,
        wandb_logger.config.project,
        f"{wandb_logger.config.exp_descriptor}_{wandb_logger.config.unique_identifier}",
    )

    # load datasets
    train_data, val_data = load_datasets()

    train_data_iter = train_data.get_iterator()
    example_batch = next(train_data_iter)

    # Example batch keys:
    # frozen_dict_keys(['observations', 'next_observations', 'actions', 'terminals', 'truncates', 'goals', 'rewards', 'masks'])
    logging.info(f"Example batch keys: {example_batch.keys()}")
    logging.info(f"Batch size: {example_batch['observations']['image'].shape[0]}")
    logging.info(f"Observation shape: {example_batch['observations']['image'].shape}")
    logging.info(f"Number of devices: {num_devices}")
    logging.info(
        f"Batch size per device: {example_batch['observations']['image'].shape[0] // num_devices}"
    )

    # we shard the leading dimension (batch dimension) accross all devices evenly
    sharding = jax.sharding.PositionalSharding(devices)
    example_batch = shard_batch(example_batch, sharding)

    # create agent
    actor_agent = create_actor_agent(example_batch)

    # replicate agent across devices
    # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
    actor_agent = jax.device_put(jax.tree_map(jnp.array, actor_agent), sharding.replicate())

    sac_rnd_config = SAC_RND_Config()

    flat_image = get_flat_obs(example_batch["observations"])[0]
    observation_mean = jnp.zeros_like(flat_image)
    observation_std = jnp.ones_like(flat_image)
    action_metadata = FLAGS.bridgedata_config.action_metadata
    action_mean, action_std = jnp.array(action_metadata["mean"]), jnp.array(
        action_metadata["std"]
    )

    key = jax.random.PRNGKey(seed=sac_rnd_config.train_seed)
    key, rnd_key, actor_key, critic_key, alpha_key = jax.random.split(key, 5)

    init_state = flat_image[None, ...]
    init_action = example_batch["actions"][0][None, ...]
    target_entropy = -init_action.shape[-1]

    rnd_module = RND(
        hidden_dim=sac_rnd_config.rnd_hidden_dim,
        embedding_dim=sac_rnd_config.rnd_embedding_dim,
        state_mean=observation_mean,
        state_std=observation_std,
        action_mean=action_mean,
        action_std=action_std,
        mlp_type=sac_rnd_config.rnd_mlp_type,
        target_mlp_type=sac_rnd_config.rnd_target_mlp_type,
        switch_features=sac_rnd_config.rnd_switch_features,
    )
    rnd = RNDTrainState.create(
        apply_fn=rnd_module.apply,
        params=rnd_module.init(rnd_key, init_state, init_action),
        tx=optax.adam(learning_rate=sac_rnd_config.rnd_learning_rate),
        rms=RunningMeanStd.create(),
    )

    alpha_module = Alpha()
    alpha = TrainState.create(
        apply_fn=alpha_module.apply,
        params=alpha_module.init(alpha_key),
        tx=optax.adam(learning_rate=sac_rnd_config.alpha_learning_rate),
    )
    critic_module = EnsembleCritic(
        hidden_dim=sac_rnd_config.hidden_dim,
        num_critics=sac_rnd_config.num_critics,
        layernorm=sac_rnd_config.critic_layernorm,
    )
    critic = CriticTrainState.create(
        apply_fn=critic_module.apply,
        params=critic_module.init(critic_key, init_state, init_action),
        target_params=critic_module.init(critic_key, init_state, init_action),
        tx=optax.adam(learning_rate=sac_rnd_config.critic_learning_rate),
    )

    update_sac_partial = partial(
        update_sac,
        target_entropy=target_entropy,
        gamma=sac_rnd_config.gamma,
        beta=sac_rnd_config.beta,
        tau=sac_rnd_config.tau,
    )

    def rnd_loop_update_step(i, carry):
        key, batch_key = jax.random.split(carry["key"])
        batch = shard_batch(next(carry["train_data"]), sharding)

        key, new_rnd, new_metrics = update_rnd(
            key, carry["rnd"], batch, carry["metrics"]
        )
        carry.update(key=key, rnd=new_rnd, metrics=new_metrics)
        return carry

    def sac_loop_update_step(i, carry):
        key, batch_key = jax.random.split(carry["key"])
        batch = shard_batch(next(carry["train_data"]), sharding)

        key, new_actor, new_critic, new_alpha, new_metrics = update_sac_partial(
            key=key,
            rnd=carry["rnd"],
            actor=carry["actor"],
            critic=carry["critic"],
            alpha=carry["alpha"],
            batch=batch,
            metrics=carry["metrics"],
        )
        carry.update(
            key=key,
            actor=new_actor,
            critic=new_critic,
            alpha=new_alpha,
            metrics=new_metrics,
        )
        return carry

    # metrics
    rnd_metrics_to_log = ["rnd_loss", "rnd_rms", "rnd_data", "rnd_random"]
    bc_metrics_to_log = [
        "critic_loss",
        "q_min",
        "actor_loss",
        "batch_entropy",
        "rnd_policy",
        "rnd_random",
        "action_mse",
        "alpha_loss",
        "alpha",
    ]
    # shared carry for update loops
    update_carry = {
        "key": key,
        "actor": actor_agent,
        "rnd": rnd,
        "critic": critic,
        "alpha": alpha,
        "train_data": train_data_iter,
    }
    # PRETRAIN RND
    for epoch in trange(sac_rnd_config.rnd_update_epochs, desc="RND Epochs", leave=True):
        # metrics for accumulation during epoch and logging to wandb, we need to reset them every epoch
        update_carry["metrics"] = Metrics.create(rnd_metrics_to_log)

        for i in trange(sac_rnd_config.num_updates_on_epoch, desc=f"Updates in Epoch {epoch}", leave=False):
            update_carry = rnd_loop_update_step(i, update_carry)

        # log mean over epoch for each metric
        mean_metrics = update_carry["metrics"].compute()
        logging.info(f"RND Epoch {epoch}: {mean_metrics}")
        wandb_logger.log(
            {"epoch": epoch, **{f"RND/{k}": v for k, v in mean_metrics.items()}}
        )

    # TRAIN BC
    for epoch in trange(sac_rnd_config.num_epochs, desc="SAC Epochs", leave=True):
        # metrics for accumulation during epoch and logging to wandb_logger, we need to reset them every epoch
        update_carry["metrics"] = Metrics.create(bc_metrics_to_log)

        for i in trange(sac_rnd_config.num_updates_on_epoch, desc=f"Updates in Epoch {epoch}", leave=False):
            update_carry = sac_loop_update_step(i, update_carry)

        # log mean over epoch for each metric
        mean_metrics = update_carry["metrics"].compute()
        logging.info(f"SAC Epoch {epoch}: {mean_metrics}")
        wandb_logger.log(
            {"epoch": epoch, **{f"SAC/{k}": v for k, v in mean_metrics.items()}}
        )

        if (epoch + 1) % FLAGS.config.eval_interval == 0:
            logging.info("Evaluating...")
            metrics = []
            val_iterator = val_data.get_iterator()

            while True:
                try:
                    batch = next(val_iterator)
                except StopIteration:
                    # No more items to iterate; break the loop
                    break
                except Exception as e:
                    logging.warning(e)
                    logging.warning("Corrupted record encountered. Skipping.")
                    continue  # Skip this iteration and continue with the next batch

                # If you reach here, you have a valid batch; process it
                rng = jax.random.PRNGKey(FLAGS.config.seed)
                rng, val_rng = jax.random.split(rng)
                metrics.append(update_carry["actor"].get_debug_metrics(batch, seed=val_rng))

            if len(metrics) > 0:
                metrics = jax.tree_map(lambda *xs: np.mean(xs), *metrics)
                logging.info(f"Validation metrics: {metrics}")
                wandb_logger.log({"validation": metrics}, step=epoch)

            del val_iterator
            
            # Explicitly clear JAX memory
            jax.device_get(jax.random.normal(jax.random.PRNGKey(0), (1,)))

            # Explicitly run Python's garbage collector
            gc.collect()

if __name__ == "__main__":
    app.run(main)
