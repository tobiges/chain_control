from cc.env import *
from cc import save, load
from cc.env.collect import collect, collect_exhaust_source,  sample_feedforward_collect_and_make_source
from cc.env.collect.collect import collect_exhaust_source, collect_random_step_source, sample_feedforward_collect_and_make_source

from cc.env.collect.source import *
from cc.env.sample_envs import TWO_SEGMENT_V1
from cc.env.wrappers import AddRefSignalRewardFnWrapper, RecordVideoWrapper
from cc.examples.pid_controller import make_pid_controller
from cc.train import *
# from experiments.helpers import calc_reward, plot_analysis
import jax.random as jrand
from cc.utils import rmse, l2_norm
import optax
import jax.numpy as jnp
from cc.examples.neural_ode_model_compact_example import make_neural_ode_model
from cc.examples.neural_ode_controller_compact_example import make_neural_ode_controller

import equinox as eqx
import pprint
import numpy
import matplotlib.pyplot as plt 
import jax as jax

import pprint
import numpy
pp = pprint.PrettyPrinter(indent=4)
numpy.set_printoptions(threshold=5)

###### Parameters#################

ref = list(range(50))


iterations = 1000

state_dim = 80
f_width_size = 0
f_depth = 0
g_width_size = 0
g_depth = 0
transform_amp = 3.0
#source_amp = 1.0
n_minibatch = 5

# Baseline:
# state_dim = 5
# f_width_size = 10
# f_depth = 0
# g_width_size = 10
# g_depth = 0

########################

env = make_env(TWO_SEGMENT_V1, random=1, time_limit=10.0, control_timestep=0.01)

model_e1 = make_neural_ode_model(
    env.action_spec(),
    env.observation_spec(),
    env.control_timestep,
    state_dim=75,
    f_depth=0,
    u_transform=jnp.arctan
)

model_e1 = eqx.tree_deserialise_leaves(
    f"/home/tobi/uni/aibe/chain_control/docs/model_e1.eqx", model_e1)



model_e2 = make_neural_ode_model(
    env.action_spec(),
    env.observation_spec(),
    env.control_timestep,
    state_dim=75,
    f_depth=0,
    u_transform=jnp.arctan
)

model_e2 = eqx.tree_deserialise_leaves(
    f"/home/tobi/uni/aibe/chain_control/docs/model_e2.eqx", model_e2)


model_e3 = make_neural_ode_model(
    env.action_spec(),
    env.observation_spec(),
    env.control_timestep,
    state_dim=75,
    f_depth=0,
    u_transform=jnp.arctan
)

model_e3 = eqx.tree_deserialise_leaves(
    f"/home/tobi/uni/aibe/chain_control/docs/model_e3.eqx", model_e3)


model_e4 = make_neural_ode_model(
    env.action_spec(),
    env.observation_spec(),
    env.control_timestep,
    state_dim=75,
    f_depth=0,
    u_transform=jnp.arctan
)

model_e4 = eqx.tree_deserialise_leaves(
    f"/home/tobi/uni/aibe/chain_control/docs/model_e4.eqx", model_e4)

source = collect_random_step_source(env, seeds=ref)

env_w_source = AddRefSignalRewardFnWrapper(env, source)


controller = make_neural_ode_controller(
    env_w_source.observation_spec(),
    env.action_spec(),
    env.control_timestep,
    state_dim=state_dim,
    f_width_size=f_width_size,
    f_depth=f_depth,
    g_width_size=g_width_size,
    g_depth=g_depth,
)

upper_bound = transform_amp
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
    n_minibatches=n_minibatch,
    tree_transform=tree_transform,
)

optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))


def our_loss_fn_reduce_along_models(log_of_loss_values):
    train_mse1 = log_of_loss_values["model_e1"]["train_mse"]
    train_mse2 = log_of_loss_values["model_e2"]["train_mse"]
    train_mse3 = log_of_loss_values["model_e3"]["train_mse"]
    train_mse4 = log_of_loss_values["model_e4"]["train_mse"]

    return {"mse_all": (train_mse1 + train_mse2 + train_mse3 + train_mse4) / 4}

controller_train_options = TrainingOptionsController(
    controller_dataloader, optimizer, 
    # loss_fn_reduce_along_models=our_loss_fn_reduce_along_models,
)

controller_trainer = ModelControllerTrainer(
    model = {
        "model_e1" : model_e1,
        # "model_e2" : model_e2,
        # "model_e3" : model_e3,
        # "model_e4" : model_e4,
    }, 
    controller=controller,
    controller_train_options=controller_train_options, 
    trackers = [Tracker("model_e1", "train_mse")],
    loggers=[DictLogger()],
)


controller_trainer.run(iterations)


eqx.tree_serialise_leaves(f"yep/controller_env_1.eqx", controller_trainer.trackers[0].best_model_or_controller())
