from cc.env import *
from cc import save, load
from cc.env.collect import collect, collect_exhaust_source,  sample_feedforward_collect_and_make_source
from cc.env.wrappers import AddRefSignalRewardFnWrapper
from cc.examples.pid_controller import make_pid_controller
from cc.train import *
import jax.random as jrand
from cc.utils import rmse, l2_norm
import optax
import jax.numpy as jnp
from cc.examples.neural_ode_model_compact_example import make_neural_ode_model
import equinox as eqx

env = make_env("two_segments_v1", random=1, time_limit=10.0, control_timestep=0.01)


model = make_neural_ode_model(
    env.action_spec(),
    env.observation_spec(),
    env.control_timestep,
    state_dim=50,
    f_depth=0, 
    u_transform=jnp.arctan
)

model = eqx.tree_deserialise_leaves("new_model.eqx", model)



source, _ = sample_feedforward_collect_and_make_source(env, seeds=[20])
env_w_source = AddRefSignalRewardFnWrapper(env, source)

controller = make_pid_controller(0.1, 0.2, 0.0, env.control_timestep)

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
controller_performance_sample = collect_exhaust_source(env_w_source, fitted_controller)

import matplotlib.pyplot as plt 

plt.plot(controller_performance_sample.obs["obs"]["xpos_of_segment_end"][0])
plt.plot(controller_performance_sample.obs["ref"]["xpos_of_segment_end"][0], label="reference")
plt.legend()
plt.savefig("out.png")