from typing import Any
from experiments.helpers import calc_reward, get_eval_source, plot_analysis

import numpy as np
import ray
from ray import air, tune

from cc.config import disable_compile_warn, disable_tqdm


# from .neural_ode_controller import make_neural_ode_controller
import equinox as eqx
from cc.env import *
from cc.env.collect import collect, collect_exhaust_source,  sample_feedforward_collect_and_make_source
from cc.env.collect.source import *
from cc.env.wrappers import AddRefSignalRewardFnWrapper
from cc.examples.pid_controller import make_pid_controller
from cc.train import *
import jax.random as jrand
from cc.utils import rmse, l2_norm
import jax
import optax
import jax.numpy as jnp
from cc.examples.neural_ode_model_compact_example import make_neural_ode_model
import pprint
import numpy
from cc.env.collect.collect import append_source, collect_random_step_source

import matplotlib.pyplot as plt 



def force_cpu_backend():
    from jax import config

    config.update("jax_platform_name", "cpu")
    config.update("jax_platforms", "cpu")


def train_and_judge(config) -> float:
    env = make_env("two_segments_v1", random=1, time_limit=10.0, control_timestep=0.01)

    model = make_neural_ode_model(
        env.action_spec(),
        env.observation_spec(),
        env.control_timestep,
        state_dim=100,
        f_depth=0,
        u_transform=jnp.arctan
    )

    model = eqx.tree_deserialise_leaves(
        "/data/ba54womo/chain_control/experiments/models/good_env2_model2.eqx", model)

    source = collect_random_step_source(env, seeds=list(range(100)), amplitude=5.0)

    #env_w_source = AddRefSignalRewardFnWrapper(env, source)

    controller = make_pid_controller(config["p"], config["i"], config["d"], env.control_timestep)

    upper_bound = 3.0
    lower_bound = -upper_bound

    @jax.vmap
    def _random_step(ref, key):
        return jnp.ones_like(ref) * jrand.uniform(
            key, (), minval=lower_bound, maxval=upper_bound
        )

    def tree_transform(key, ref, bs):
        keys = jrand.split(key, bs)
        return jtu.tree_map(lambda ref: _random_step(ref, keys), ref)

    controller_dataloader = make_dataloader(
        UnsupervisedDataset(source.get_references_for_optimisation()),
         jrand.PRNGKey(1,),
         n_minibatches=5,
         tree_transform=tree_transform,
    )


    controller_train_options = TrainingOptionsController(
        controller_dataloader, optax.adam(3e-3), 
    )

    controller_trainer = ModelControllerTrainer(
        model, controller, controller_train_options=controller_train_options, 
        trackers=[Tracker("train_mse")]
    )

    controller_trainer.run(500)

    fitted_controller = controller_trainer.trackers[0].best_model_or_controller()

    eval_source = get_eval_source(5, 5, 5)
    controller_performance_sample = collect_exhaust_source(
        AddRefSignalRewardFnWrapper(env, eval_source), fitted_controller)

    plot_analysis(fitted_controller, env, model, f"images/plot_state:p_{config['p']}_i:{config['i']}_d:{config['d']}.png", False)


    return calc_reward(controller_performance_sample)

config = {
    "p": 1.0,
    "i": 0.45,
    "d": -0.67,
}
force_cpu_backend()
print(train_and_judge(config))