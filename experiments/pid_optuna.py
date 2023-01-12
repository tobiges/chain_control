from typing import Any
from experiments.helpers import calc_reward, get_eval_source, plot_analysis

import numpy as np

import optuna

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
import optax
import jax.numpy as jnp
import pprint
import numpy as np
from cc.examples.neural_ode_model_compact_example import make_neural_ode_model

from cc.env.collect.collect import append_source, collect_random_step_source
#from cc.examples.neural_ode_controller_compact_example import make_neural_ode_controller
from controllers.nonlinear_ode import make_neural_ode_controller

import matplotlib.pyplot as plt


def force_cpu_backend():
    from jax import config

    config.update("jax_platform_name", "cpu")
    config.update("jax_platforms", "cpu")


def objective(trial) -> float:
    config = {
        "p": trial.suggest_float('p', -2, 2),
        "i": trial.suggest_float('i', -2, 2),
        "d": trial.suggest_float('d', -2, 2),
    }

    env = make_env("two_segments_v1", random=1, time_limit=10.0, control_timestep=0.01)

    model = make_neural_ode_model(
        env.action_spec(),
        env.observation_spec(),
        env.control_timestep,
        state_dim=50,
        f_depth=0,
        u_transform=jnp.arctan
    )

    model = eqx.tree_deserialise_leaves(
        "/data/ba54womo/chain_control/experiments/models/good_env1_model.eqx", model)

    source, _ = sample_feedforward_collect_and_make_source(env, seeds=list(range(20, 30)))
    # source = constant_after_transform_source(source, after_T = 3.0)

    env_w_source = AddRefSignalRewardFnWrapper(env, source)

    controller = make_pid_controller(config["p"], config["i"], config["d"], env.control_timestep)

    controller_dataloader = make_dataloader(
         source.get_references_for_optimisation(),
         jrand.PRNGKey(1,),
         n_minibatches=2
    )

    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))


    controller_train_options = TrainingOptionsController(
        controller_dataloader, optimizer, 
    )

    controller_trainer = ModelControllerTrainer(
        model, controller, controller_train_options=controller_train_options, 
        trackers=[Tracker("train_mse")]
    )

    controller_trainer.run(500)

    eval_source = get_eval_source(5, 0, 0)

    fitted_controller = controller_trainer.trackers[0].best_model_or_controller()
    controller_performance_sample = collect_exhaust_source(
        AddRefSignalRewardFnWrapper(env, eval_source), fitted_controller)

    # plt.plot(controller_performance_sample.obs["obs"]["xpos_of_segment_end"][0], label=f"observation")
    # plt.plot(controller_performance_sample.obs["ref"]["xpos_of_segment_end"][0], label="reference")
    # plt.legend()
    # plt.savefig(f"out_{config['p']}_{config['i']}_{config['d']}.png")
    # plt.clf()

    plot_analysis(fitted_controller, env, model, f"optuna/{trial.study.study_name}/trial:{trial.number}_p:{config['p']}_i:{config['i']}_d:{config['d']}.png")

    return np.sum(np.abs(controller_performance_sample.rew))


# search_space = {
#     "state_dim": tune.grid_search([80]),
#     "f_width_size": tune.grid_search([0]),
#     "f_depth": tune.grid_search([0]),
#     "g_width_size": tune.grid_search([0]),
#     "g_depth": tune.grid_search([0]),
#     "u_transform": tune.grid_search([lambda u: u]),
# }

force_cpu_backend()
# Otherwise the stdout will be messy
disable_tqdm()
disable_compile_warn()

study = optuna.load_study(storage="sqlite:///optuna.db", study_name="pid_study")
study.optimize(objective, n_trials=10)
print(study.best_params)
print(study.best_value)
print(study.best_trial)
