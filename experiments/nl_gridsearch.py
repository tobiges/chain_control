from typing import Any
from experiments.helpers import calc_reward, get_eval_source, plot_analysis

import numpy as np
import ray
from ray import air, tune

from cc.config import disable_compile_warn, disable_tqdm

import equinox as eqx
from cc.env import *
from cc.env.collect import collect, collect_exhaust_source,  sample_feedforward_collect_and_make_source
from cc.env.collect.source import *
from cc.env.wrappers import AddRefSignalRewardFnWrapper
from cc.examples.pid_controller import make_pid_controller
from cc.train import *
import jax.random as jrand
from cc.utils import rmse, l2_norm
import optax
import jax.numpy as jnp
import pprint
import numpy as np
from cc.examples.neural_ode_model_compact_example import make_neural_ode_model

from cc.env.collect.collect import append_source, collect_random_step_source

from controllers.nonlinear_ode import make_neural_ode_controller

import matplotlib.pyplot as plt 

def force_cpu_backend():
    from jax import config

    config.update("jax_platform_name", "cpu")
    config.update("jax_platforms", "cpu")


def train_and_judge(config) -> float:

    # Skip invalid
    if config["f_depth"] != 0 and config["f_width_size"] == 0 or config["g_depth"] != 0 and config["g_width_size"] == 0:
        return 123456789.0
    if config["f_depth"] == 0 and config["f_width_size"] > 0 or config["g_depth"] == 0 and config["g_width_size"] > 0:
        return 234567890.0

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
        "/data/ba54womo/chain_control/experiments/models/good_env1_model2.eqx", model)

    source = collect_random_step_source(env, seeds=list(range(3000)), amplitude=5.0)

    env_w_source = AddRefSignalRewardFnWrapper(env, source)

    controller = make_neural_ode_controller(
        env_w_source.observation_spec(),
        env.action_spec(),
        env.control_timestep,
        state_dim=config["state_dim"],
        f_width_size=config["f_width_size"],
        f_depth=config["f_depth"],
        g_width_size=config["g_width_size"],
        g_depth=config["g_depth"],
        u_transform=lambda u:u
    )

    controller_dataloader = make_dataloader(
        UnsupervisedDataset(source.get_references_for_optimisation()),
        jrand.PRNGKey(1,),
        n_minibatches=5
    )

    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))

    controller_train_options = TrainingOptionsController(
        controller_dataloader, optimizer, )

    controller_trainer = ModelControllerTrainer(
        model, controller, controller_train_options=controller_train_options,
        trackers=[Tracker("train_mse")]
    )

    controller_trainer.run(100)

    fitted_controller = controller_trainer.trackers[0].best_model_or_controller()

    eval_source = get_eval_source(5, 5, 5)
    controller_performance_sample = collect_exhaust_source(
        AddRefSignalRewardFnWrapper(env, eval_source), fitted_controller)

    plot_analysis(fitted_controller, env, model, f"plot_state:{config['state_dim']}_fd:{config['f_depth']}_fw:{config['f_width_size']}_gd:{config['g_depth']}_fw:{config['g_width_size']}.png", False)


    return calc_reward(controller_performance_sample)


def objective(config):
    force_cpu_backend()
    # Otherwise the stdout will be messy
    disable_tqdm()
    disable_compile_warn()

    return train_and_judge(config)


ray.init()

search_space = {
    "state_dim": tune.grid_search([30, 50, 80, 100]),
    "f_width_size": tune.grid_search([0, 1, 5, 10, 30]),
    "f_depth": tune.grid_search([0, 1, 2]),
    "g_width_size": tune.grid_search([0, 1, 5, 10, 30]),
    "g_depth": tune.grid_search([0, 1, 2]),
    "u_transform": tune.grid_search([lambda u: u]),
}

tuner = tune.Tuner(
    tune.with_resources(objective, {"cpu": 1}),
    tune_config=tune.TuneConfig(
        num_samples=1,  # <- 1 sample *per* grid point
        time_budget_s=24 * 3600 * 14,  # maximum runtime in seconds
        mode="min",  # either `min` or `max`
    ),
    run_config=air.RunConfig(
        log_to_file=("my_stdout.log", "my_stderr.log"), local_dir="ray_results"
    ),
    param_space=search_space,
)

#tuner = tune.Tuner.restore(
#    path="/data/ba54womo/chain_control/experiments/ray_results/objective_2022-11-29_16-17-26",
#    resume_errored=True
#)
tuner.fit()

ray.shutdown()
