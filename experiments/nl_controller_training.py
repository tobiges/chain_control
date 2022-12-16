from cc.env import *
from cc import save, load
from cc.env.collect import collect, collect_exhaust_source,  sample_feedforward_collect_and_make_source
from cc.env.collect.collect import append_source, collect_exhaust_source, collect_random_step_source, sample_feedforward_collect_and_make_source

from cc.env.collect.source import *
from cc.env.wrappers import AddRefSignalRewardFnWrapper, RecordVideoWrapper
from cc.examples.neural_ode_controller_compact_example import make_neural_ode_controller
from cc.examples.pid_controller import make_pid_controller
from cc.train import *
from experiments.helpers import calc_reward, plot_analysis
import jax.random as jrand
from cc.utils import rmse, l2_norm
import optax
import jax.numpy as jnp
from cc.examples.neural_ode_model_compact_example import make_neural_ode_model
import equinox as eqx
import pprint
import numpy
import matplotlib.pyplot as plt 
import jax as jax

pp = pprint.PrettyPrinter(indent=4)
numpy.set_printoptions(threshold=5)

###### Parameters#################

model_num = ""
env_num = 1

ref = list(range(5000))

constant_after = 3.0

iterations = 500

state_dim = 80
f_width_size = 0
f_depth = 0
g_width_size = 0
g_depth = 0

# Baseline:
# state_dim = 5
# f_width_size = 10
# f_depth = 0
# g_width_size = 10
# g_depth = 0

########################

env1 = make_env(f"two_segments_v{env_num}", random=1, time_limit=10.0, control_timestep=0.01)


model1 = make_neural_ode_model(
    env1.action_spec(),
    env1.observation_spec(),
    env1.control_timestep,
    state_dim=100,
    f_depth=0,
    u_transform=jnp.arctan
)

model1 = eqx.tree_deserialise_leaves(
    f"/data/ba54womo/chain_control/experiments/models/good_env{env_num}_model{model_num}.eqx", model1)

#source1, _ = sample_feedforward_collect_and_make_source(env1, seeds=[ref])
#source1 = constant_after_transform_source(source1, after_T = constant_after, new_ts = env1.ts)

source1 = collect_random_step_source(env1, seeds=ref, amplitude=5.0)

env1_w_source = AddRefSignalRewardFnWrapper(env1, source1)


controller1 = make_neural_ode_controller(
    env1_w_source.observation_spec(),
    env1.action_spec(),
    env1.control_timestep,
    state_dim=state_dim,
    f_width_size=f_width_size,
    f_depth=f_depth,
    g_width_size=g_width_size,
    g_depth=g_depth,
)

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

controller_dataloader1 = make_dataloader(
    UnsupervisedDataset(source1.get_references_for_optimisation()),
    jrand.PRNGKey(1,),
    n_minibatches=5,
    tree_transform=tree_transform,
)
optimizer1 = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))
controller_train_options1 = TrainingOptionsController(
    controller_dataloader1, optimizer1, 
)
controller_trainer1 = ModelControllerTrainer(
    model1, controller1, controller_train_options=controller_train_options1, 
    trackers=[Tracker("train_mse")],
    loggers=[DictLogger()]
)
controller_trainer1.run(iterations)

fitted_controller1 = controller_trainer1.trackers[0].best_model_or_controller()
# env1_w_video = RecordVideoWrapper(env1_w_source, width=1280, height=720, cleanup_imgs=False)

plot_analysis(fitted_controller1, env1, model1, f"images/plot_env{env_num}_{len(ref)}ref_{iterations}it.png", f"two_segments_v{env_num}")

#plt.plot(controller_performance_sample1.obs["obs"]["xpos_of_segment_end"][0], label="obs")
#plt.plot(controller_performance_sample1.obs["ref"]["xpos_of_segment_end"][0], label="reference")
#plt.legend()
#plt.savefig(f"images/nl_{state_dim}_{f_depth}_{f_width_size}_{g_depth}_{g_width_size}_ref{ref}_v{env_num}_ca{constant_after}_it{iterations}.png")
#
#print(np.sum(np.abs(controller_performance_sample1.rew)))
#print(np.sum(np.abs(controller_performance_sample1.rew))/len(controller_performance_sample1.rew[0]))
