import jax
import jax.numpy as jnp
import optax

from flax.core import FrozenDict
from typing import Dict, Tuple, Any
from tqdm.auto import trange

from flax.training.train_state import TrainState
from jaxrl_m.common.common import JaxRLTrainState
from sac_rnd.offline_sac.utils.common import Metrics

from sac_rnd.offline_sac.utils.running_moments import RunningMeanStd

# import os
# import sys
# current_path = os.path.dirname(os.path.abspath(__file__))
# sac_rnd_path = os.path.join(current_path, "sac_rnd")
# sys.path.append(sac_rnd_path)


class RNDTrainState(TrainState):
    rms: RunningMeanStd


class CriticTrainState(TrainState):
    target_params: FrozenDict


def get_flat_obs(obs) -> jax.Array:
    obs_shape = obs["image"].shape
    return obs["image"].reshape(obs_shape[0], -1)


# RND functions
def rnd_bonus(rnd: RNDTrainState, state: jax.Array, action: jax.Array) -> jax.Array:
    pred, target = rnd.apply_fn(rnd.params, state, action)
    # [batch_size, embedding_dim]
    bonus = jnp.sum((pred - target) ** 2, axis=1) / rnd.rms.std
    return bonus


def update_rnd(
    key: jax.random.PRNGKey,
    rnd: RNDTrainState,
    batch: Dict[str, jax.Array],
    metrics: Metrics,
) -> Tuple[jax.random.PRNGKey, RNDTrainState, Metrics]:
    def rnd_loss_fn(params):
        pred, target = rnd.apply_fn(
            params, get_flat_obs(batch["observations"]), batch["actions"]
        )
        raw_loss = ((pred - target) ** 2).sum(axis=1)

        new_rms = rnd.rms.update(raw_loss)
        loss = raw_loss.mean(axis=0)
        return loss, new_rms

    (loss, new_rms), grads = jax.value_and_grad(rnd_loss_fn, has_aux=True)(rnd.params)
    new_rnd = rnd.apply_gradients(grads=grads).replace(rms=new_rms)

    # log rnd bonus for random actions
    key, actions_key = jax.random.split(key)
    random_actions = jax.random.uniform(
        actions_key, shape=batch["actions"].shape, minval=-1.0, maxval=1.0
    )
    new_metrics = metrics.update(
        {
            "rnd_loss": loss,
            "rnd_rms": new_rnd.rms.std,
            "rnd_data": loss / rnd.rms.std,
            "rnd_random": rnd_bonus(
                rnd, get_flat_obs(batch["observations"]), random_actions
            ).mean(),
        }
    )
    return key, new_rnd, new_metrics


def update_actor(
    key: jax.random.PRNGKey,
    actor: JaxRLTrainState,
    rnd: RNDTrainState,
    critic: TrainState,
    alpha: TrainState,
    batch: Dict[str, jax.Array],
    beta: float,
    metrics: Metrics,
) -> Tuple[jax.random.PRNGKey, TrainState, jax.Array, Metrics]:
    key, actions_key, random_action_key = jax.random.split(key, 3)

    def actor_loss_fn(params):
        actions_dist = actor.apply_fn(
            {"params": params},
            (batch["observations"], batch["goals"]),
            temperature=1.0,
            train=True,
            rngs={"dropout": key},
            name="actor",
        )
        actions, actions_logp = actions_dist.sample_and_log_prob(seed=actions_key)

        rnd_penalty = rnd_bonus(rnd, get_flat_obs(batch["observations"]), actions)
        q_values = critic.apply_fn(
            critic.params, get_flat_obs(batch["observations"]), actions
        ).min(0)
        loss = (
            alpha.apply_fn(alpha.params) * actions_logp.sum(-1)
            + beta * rnd_penalty
            - q_values
        ).mean()

        # logging stuff
        actor_entropy = -actions_logp.sum(-1).mean()
        random_actions = jax.random.uniform(
            random_action_key, shape=batch["actions"].shape, minval=-1.0, maxval=1.0
        )
        new_metrics = metrics.update(
            {
                "batch_entropy": actor_entropy,
                "actor_loss": loss,
                "rnd_policy": rnd_penalty.mean(),
                "rnd_random": rnd_bonus(
                    rnd, get_flat_obs(batch["observations"]), random_actions
                ).mean(),
                "action_mse": ((actions - batch["actions"]) ** 2).mean(),
            }
        )
        return loss, (actor_entropy, new_metrics)

    grads, (actor_entropy, new_metrics) = jax.grad(actor_loss_fn, has_aux=True)(
        actor.params
    )
    new_actor = actor.apply_gradients(grads=grads)

    return key, new_actor, actor_entropy, new_metrics


def update_alpha(
    alpha: TrainState, entropy: jax.Array, target_entropy: float, metrics: Metrics
) -> Tuple[TrainState, Metrics]:
    def alpha_loss_fn(params):
        alpha_value = alpha.apply_fn(params)
        loss = (alpha_value * (entropy - target_entropy)).mean()

        new_metrics = metrics.update({"alpha": alpha_value, "alpha_loss": loss})
        return loss, new_metrics

    grads, new_metrics = jax.grad(alpha_loss_fn, has_aux=True)(alpha.params)
    new_alpha = alpha.apply_gradients(grads=grads)

    return new_alpha, new_metrics


def update_critic(
    key: jax.random.PRNGKey,
    actor: TrainState,
    rnd: RNDTrainState,
    critic: CriticTrainState,
    alpha: TrainState,
    batch: Dict[str, jax.Array],
    gamma: float,
    beta: float,
    tau: float,
    metrics: Metrics,
) -> Tuple[jax.random.PRNGKey, TrainState, Metrics]:
    key, actions_key = jax.random.split(key)

    next_actions_dist = actor.apply_fn(
        {"params": actor.params},
        (batch["next_observations"], batch["next_goals"]),
        temperature=1.0,
        train=False,
        rngs={"dropout": key},
        name="actor",
    )

    next_actions, next_actions_logp = next_actions_dist.sample_and_log_prob(
        seed=actions_key
    )
    rnd_penalty = rnd_bonus(rnd, get_flat_obs(batch["next_observations"]), next_actions)

    next_q = critic.apply_fn(
        critic.target_params, get_flat_obs(batch["next_observations"]), next_actions
    ).min(0)
    next_q = (
        next_q
        - alpha.apply_fn(alpha.params) * next_actions_logp.sum(-1)
        - beta * rnd_penalty
    )

    target_q = batch["rewards"] + (1 - batch["truncates"]) * gamma * next_q

    def critic_loss_fn(critic_params):
        # [N, batch_size] - [1, batch_size]
        q = critic.apply_fn(
            critic_params, get_flat_obs(batch["observations"]), batch["actions"]
        )
        q_min = q.min(0).mean()
        loss = ((q - target_q[None, ...]) ** 2).mean(1).sum(0)
        return loss, q_min

    (loss, q_min), grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(
        critic.params
    )
    new_critic = critic.apply_gradients(grads=grads)
    new_critic = new_critic.replace(
        target_params=optax.incremental_update(
            new_critic.params, new_critic.target_params, tau
        )
    )
    new_metrics = metrics.update(
        {
            "critic_loss": loss,
            "q_min": q_min,
        }
    )
    return key, new_critic, new_metrics


def update_sac(
    key: jax.random.PRNGKey,
    rnd: RNDTrainState,
    actor: TrainState,
    critic: CriticTrainState,
    alpha: TrainState,
    batch: Dict[str, Any],
    target_entropy: float,
    gamma: float,
    beta: float,
    tau: float,
    metrics: Metrics,
):
    key, new_actor, actor_entropy, new_metrics = update_actor(
        key, actor, rnd, critic, alpha, batch, beta, metrics
    )
    new_alpha, new_metrics = update_alpha(
        alpha, actor_entropy, target_entropy, new_metrics
    )
    key, new_critic, new_metrics = update_critic(
        key, new_actor, rnd, critic, alpha, batch, gamma, beta, tau, new_metrics
    )
    return key, new_actor, new_critic, new_alpha, new_metrics
