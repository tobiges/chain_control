from cc.env import *
from cc import save, load
from cc.env.collect import collect, collect_exhaust_source,  sample_feedforward_collect_and_make_source
from cc.env.collect.source import *
from cc.env.wrappers import AddRefSignalRewardFnWrapper, RecordVideoWrapper
from cc.examples.pid_controller import make_pid_controller
from cc.train import *
import jax.random as jrand
from cc.utils import rmse, l2_norm
import optax
import jax.numpy as jnp
from cc.examples.neural_ode_model_compact_example import make_neural_ode_model
import equinox as eqx
import pprint
import numpy
pp = pprint.PrettyPrinter(indent=4)
numpy.set_printoptions(threshold=5)

env_num = 2

env = make_env(f"two_segments_v{env_num}", random=1, time_limit=10.0, control_timestep=0.01)


model = make_neural_ode_model(
    env.action_spec(),
    env.observation_spec(),
    env.control_timestep,
    state_dim=12,
    f_integrate_method="EE",
    f_depth=2,
    f_width_size=25,
    g_depth=0,
    u_transform=jnp.arctan
)

p, i, d = 10.0, -4.0, 2.0
# p, i, d = 0.1, 0.1, 0.0

ref = 20

model = eqx.tree_deserialise_leaves("created_model.eqx", model)

source, _ = sample_feedforward_collect_and_make_source(env, seeds=[ref])
source = constant_after_transform_source(source, after_T = 3.0)

env_w_source = AddRefSignalRewardFnWrapper(env, source)

controller = make_pid_controller(p, i, d, env.control_timestep)

controller_dataloader = make_dataloader(
    source.get_references_for_optimisation(),
    jrand.PRNGKey(1,),
    n_minibatches=1
)


controller_train_options = TrainingOptionsController(
    controller_dataloader, optax.adam(3e-3), 
)

controller_trainer = ModelControllerTrainer(
    model, controller, controller_train_options=controller_train_options, 
    trackers=[Tracker("train_mse")]
)

controller_trainer.run(500)

fitted_controller = controller_trainer.trackers[0].best_model()
env_w_video = RecordVideoWrapper(env_w_source, width=1280, height=720, cleanup_imgs=False)
controller_performance_sample = collect_exhaust_source(env_w_video, fitted_controller)

pp.pprint(controller_performance_sample)

import matplotlib.pyplot as plt 

plt.plot(controller_performance_sample.obs["obs"]["xpos_of_segment_end"][0], label="observation")
plt.plot(controller_performance_sample.obs["ref"]["xpos_of_segment_end"][0], label="reference")
plt.legend()
plt.savefig(f"images/out_{p}_{i}_{d}_ref{ref}_v{env_num}.png")

print(np.sum(np.abs(controller_performance_sample.rew)))
