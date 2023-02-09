# %%
import equinox as eqx
import numpy as np
from cc.env.collect import sample_feedforward_and_collect
from cc.env import make_env
from cc.env.envs.two_segments import CartParams, JointParams, generate_env_config
from cc.examples.neural_ode_model_compact_example import make_neural_ode_model
from cc.train import (
    ModelControllerTrainer, TrainingOptionsModel,
    EvaluationMetrices, Tracker, make_dataloader, DictLogger,
    Regularisation, SupervisedDataset
)
import jax.random as jrand
from cc.utils import rmse, l2_norm
import optax
import jax.numpy as jnp

# %%
time_limit = 10.0
control_timestep = 0.01

env1 = generate_env_config(CartParams(
    name="cart",
    slider_joint_params=JointParams(damping=1e-3),
    hinge_joint_params=JointParams(
        damping=1e-1, springref=0, stiffness=10
    ),
))

env2 = generate_env_config(CartParams(
    name="cart",
    slider_joint_params=JointParams(damping=1e-3),
    hinge_joint_params=JointParams(
        damping=1e-1, springref=0, stiffness=2
    ),
))

env3 = generate_env_config(CartParams(
    name="cart",
    slider_joint_params=JointParams(damping=1e-3),
    hinge_joint_params=JointParams(
        damping=3e-2, springref=0, stiffness=10
    ),
))

env4 = generate_env_config(CartParams(
    name="cart",
    slider_joint_params=JointParams(damping=1e-3),
    hinge_joint_params=JointParams(
        damping=3e-2, springref=0, stiffness=2
    ),
))


env = make_env(env3, random=1)

# %%
sample_train = sample_feedforward_and_collect(
    env,
    seeds_gp=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    seeds_cos=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14]
)

sample_val = sample_feedforward_and_collect(
    env,
    seeds_gp=[15, 16, 17, 18],
    seeds_cos=[2.5, 5.0, 7.5, 10.0]
)

model = make_neural_ode_model(
    env.action_spec(),
    env.observation_spec(),
    env.control_timestep,
    75,
    f_depth=0,
    u_transform=jnp.arctan,
)

model_train_dataloader = make_dataloader(
    SupervisedDataset(sample_train.action, sample_train.obs),  # <- (X, y)
    jrand.PRNGKey(2,), 
    n_minibatches=4
)

optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))

metrices = (
    EvaluationMetrices(
        data = (sample_val.action, sample_val.obs), # <- (X, y)
        metrices=(
            lambda y, yhat: {"val_rmse": rmse(y, yhat)}, 
        )
    ),
)

model_train_options = TrainingOptionsModel(
    model_train_dataloader, 
    optimizer, 
    metrices=metrices
)

model_trainer = ModelControllerTrainer(
    model, 
    model_train_options=model_train_options,
    loggers=[DictLogger()],
    trackers=[Tracker("val_rmse")]
)

model_trainer.run(1000)

# %%
# model = make_neural_ode_model(
#     env.action_spec(),
#     env.observation_spec(),
#     env.control_timestep,
#     state_dim=75,
#     f_depth=0,
#     u_transform=jnp.arctan
# )

# %%
# model_train_dataloader = make_dataloader(
#     SupervisedDataset(sample_train.action, sample_train.obs),  # <- (X, y)
#     jrand.PRNGKey(2,),
#     n_minibatches=4
# )

# optimizer = optax.adam(1e-3)

# regularisers = (
#     Regularisation(
#         prefactor=0.5,
#         reduce_weights=lambda vector_of_params: {"l2_norm": l2_norm(vector_of_params)}
#     ),
# )

# metrices = (
#     EvaluationMetrices(
#         data=(sample_val.action, sample_val.obs),  # <- (X, y)
#         metrices=(
#             lambda y, yhat: {"val_rmse": rmse(y, yhat)},
#         )
#     ),
# )

# model_train_options = TrainingOptionsModel(
#     model_train_dataloader,
#     optimizer,
#     regularisers=regularisers,
#     metrices=metrices
# )

# model_trainer = ModelControllerTrainer(
#     model,
#     model_train_options=model_train_options,
#     loggers=[DictLogger()],
#     trackers=[Tracker("val_rmse")]
# )

# # %%
# model_trainer.run(1000)

# %%
model_trainer.trackers[0].best_metric()

# %%
fitted_model = model_trainer.trackers[0].best_model_or_controller()

# save model for usage in next notebook
eqx.tree_serialise_leaves(f"model_new_e3.eqx", fitted_model)
