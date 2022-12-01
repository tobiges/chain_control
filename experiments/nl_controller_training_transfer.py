from cc.env import *
from cc import save, load
from cc.env.collect import collect, collect_exhaust_source,  sample_feedforward_collect_and_make_source
from cc.env.collect.source import *
from cc.env.wrappers import AddRefSignalRewardFnWrapper, RecordVideoWrapper
from cc.examples.neural_ode_controller_compact_example import make_neural_ode_controller
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
import matplotlib.pyplot as plt 

pp = pprint.PrettyPrinter(indent=4)
numpy.set_printoptions(threshold=5)

## Parameters

env_num = 1

ref = 20

constant_after = 3.0

iterations = 500

state_dim = 5
f_width_size = 1
f_depth = 25
g_width_size = 25
g_depth = 10

#######

env1 = make_env(f"two_segments_v1", random=1, time_limit=10.0, control_timestep=0.01)
env2 = make_env(f"two_segments_v2", random=1, time_limit=10.0, control_timestep=0.01)


model1 = make_neural_ode_model(
    env1.action_spec(),
    env1.observation_spec(),
    env1.control_timestep,
    state_dim=12,
    f_integrate_method="EE",
    f_depth=2,
    f_width_size=25,
    g_depth=0,
    u_transform=jnp.arctan
)
model2 = make_neural_ode_model(
    env2.action_spec(),
    env2.observation_spec(),
    env2.control_timestep,
    state_dim=12,
    f_integrate_method="EE",
    f_depth=2,
    f_width_size=25,
    g_depth=0,
    u_transform=jnp.arctan
)

model1 = eqx.tree_deserialise_leaves("env1_model.eqx", model1)
model2 = eqx.tree_deserialise_leaves("env2_model.eqx", model2)

source1, _ = sample_feedforward_collect_and_make_source(env1, seeds=[ref])
source1 = constant_after_transform_source(source1, after_T = constant_after)
env1_w_source = AddRefSignalRewardFnWrapper(env1, source1)

source2, _ = sample_feedforward_collect_and_make_source(env1, seeds=[ref])
source2 = constant_after_transform_source(source2, after_T = constant_after)
env2_w_source = AddRefSignalRewardFnWrapper(env2, source2)


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
controller_dataloader1 = make_dataloader(
    source1.get_references_for_optimisation(),
    jrand.PRNGKey(1,),
    n_minibatches=1
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


controller2 = make_neural_ode_controller(
    env2_w_source.observation_spec(),
    env2.action_spec(),
    env2.control_timestep,
    state_dim=state_dim,
    f_width_size=f_width_size,
    f_depth=f_depth,
    g_width_size=g_width_size,
    g_depth=g_depth,
)
controller_dataloader2 = make_dataloader(
    source2.get_references_for_optimisation(),
    jrand.PRNGKey(1,),
    n_minibatches=1
)
optimizer2 = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))
controller_train_options2 = TrainingOptionsController(
    controller_dataloader2, optimizer2, 
)
controller_trainer2 = ModelControllerTrainer(
    model2, controller2, controller_train_options=controller_train_options2, 
    trackers=[Tracker("train_mse")],
    loggers=[DictLogger()]
)
controller_trainer2.run(iterations)






# pp.pprint(controller_trainer.loggers[0].get_logs())

# plt.plot(controller_trainer.loggers[0].get_logs()["train_loss"], label="train_loss")
# plt.plot(controller_trainer.loggers[0].get_logs()["train_mse"], label="train_mse")
# plt.legend()
# plt.savefig(f"nl_train_losses.png")

# env_w_video = RecordVideoWrapper(env_w_source, width=1280, height=720, cleanup_imgs=False)

fitted_controller1 = controller_trainer1.trackers[0].best_model()
controller_performance_sample1 = collect_exhaust_source(env1_w_source, fitted_controller1)

fitted_controller2 = controller_trainer2.trackers[0].best_model()
controller_performance_sample2 = collect_exhaust_source(env2_w_source, fitted_controller2)

controller_performance_sample1_in_2 = collect_exhaust_source(env2_w_source, fitted_controller1)
controller_performance_sample2_in_1 = collect_exhaust_source(env1_w_source, fitted_controller2)

# pp.pprint(controller_performance_sample)


plt.plot(controller_performance_sample1.obs["obs"]["xpos_of_segment_end"][0], label="controller trained in env 1")
plt.plot(controller_performance_sample2_in_1.obs["obs"]["xpos_of_segment_end"][0], label="controller trained in env 2")
plt.plot(controller_performance_sample1.obs["ref"]["xpos_of_segment_end"][0], label="reference")
plt.legend()
# plt.savefig(f"images/nl_{state_dim}_{f_depth}_{f_width_size}_{g_depth}_{g_width_size}_ref{ref}_v{env_num}_ca{constant_after}_it{iterations}.png")
plt.savefig(f"images/env1_transfer.png")
plt.clf()

plt.plot(controller_performance_sample2.obs["obs"]["xpos_of_segment_end"][0], label="controller trained in env 2")
plt.plot(controller_performance_sample1_in_2.obs["obs"]["xpos_of_segment_end"][0], label="controller trained in env 1")
plt.plot(controller_performance_sample2.obs["ref"]["xpos_of_segment_end"][0], label="reference")
plt.legend()
plt.savefig(f"images/env2_transfer.png")
plt.clf()

# print(np.sum(np.abs(controller_performance_sample1.rew)))
# print(np.sum(np.abs(controller_performance_sample1.rew))/len(controller_performance_sample1.rew[0]))
