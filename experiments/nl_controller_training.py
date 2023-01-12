from cc.env import *
from cc import save, load
from cc.env.collect import collect, collect_exhaust_source,  sample_feedforward_collect_and_make_source
from cc.env.collect.collect import append_source, collect_exhaust_source, collect_random_step_source, sample_feedforward_collect_and_make_source

from cc.env.collect.source import *
from cc.env.wrappers import AddRefSignalRewardFnWrapper, RecordVideoWrapper
from cc.examples.pid_controller import make_pid_controller
from cc.train import *
from experiments.helpers import calc_reward, plot_analysis
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

env_num = 1
eval_env_num = 1

ref = list(range(20,27))

constant_after = 3.0

iterations = 3000

state_dim = 5
f_width_size = 10
f_depth = 1
g_width_size = 0
g_depth = 0
transform_amp = 5.0
source_amp = 1.0
n_minibatch = 1

# Baseline:
# state_dim = 5
# f_width_size = 10
# f_depth = 0
# g_width_size = 10
# g_depth = 0

########################

env = make_env(f"two_segments_v{env_num}", random=1, time_limit=10.0, control_timestep=0.01)

model = make_neural_ode_model(
    env.action_spec(),
    env.observation_spec(),
    env.control_timestep,
    state_dim=50,
    f_depth=0,
    u_transform=jnp.arctan
)

model = eqx.tree_deserialise_leaves(
    f"/data/ba54womo/chain_control/experiments/models/good_env1_model.eqx", model)

source, _, _ = sample_feedforward_collect_and_make_source(env, seeds=ref)
source = constant_after_transform_source(source, after_time = constant_after)

#source = collect_random_step_source(env, seeds=ref, amplitude=source_amp)

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

controller_train_options = TrainingOptionsController(
    controller_dataloader, optimizer, 
)
controller_trainer = ModelControllerTrainer(
    model = {"model" : model}, controller= controller, controller_train_options=controller_train_options, 
    trackers = [Tracker("model", "train_mse")],
    loggers=[DictLogger()],
)


controller_trainer.run(iterations)


fitted_controller = controller_trainer.trackers[0].best_model_or_controller()


plot_analysis(fitted_controller, env, model, f"images/{eval_env_num}_trainenv{env_num}_ref{len(ref)}_state{state_dim}_it{iterations}_mini{n_minibatch}.png",
                                             False, f"two_segments_v{eval_env_num}")
