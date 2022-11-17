from cc.env import *
from cc import save, load
from cc.env.collect import collect, collect_exhaust_source,  sample_feedforward_collect_and_make_source
from cc.env.collect import sample_feedforward_and_collect
from cc.env.wrappers import AddRefSignalRewardFnWrapper
from cc.examples.pid_controller import make_pid_controller
from cc.train import *
import jax.random as jrand
from cc.utils import rmse, l2_norm
import optax
import jax.numpy as jnp
from cc.examples.neural_ode_model_compact_example import make_neural_ode_model
import equinox as eqx

time_limit = 10.0
control_timestep = 0.01

env = make_env("two_segments_v1", time_limit=time_limit, control_timestep=control_timestep, random=1)

sample_train = sample_feedforward_and_collect(
    env,
   seeds_gp=[0,1,2,4,5,6,7,8,9,10,11,12,13,14],
    seeds_cos=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
)

sample_val = sample_feedforward_and_collect(
    env, 
    seeds_gp=[15, 16, 17, 18, 19],
    seeds_cos=[2.5, 5.0, 7.5, 10.0, 12.5]
)

model = make_neural_ode_model(
    env.action_spec(),
    env.observation_spec(),
    env.control_timestep,
    state_dim=12,
    f_integrate_method="EE",
    f_depth=2,
    f_width_size=25,
    g_depth=0
)

model_train_dataloader = make_dataloader(
    (sample_train.action, sample_train.obs), # <- (X, y)
    jrand.PRNGKey(1,), 
    n_minibatches=1
)

optimizer = optax.adam(3e-3)

regularisers = (
    Regularisation(
        prefactor = 0.05,
        reduce_weights = lambda vector_of_params: {"l2_norm": l2_norm(vector_of_params)}
    ),
)

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
    regularisers=(),
    metrices=metrices
)

model_trainer = ModelControllerTrainer(
    model, 
    model_train_options=model_train_options,
    loggers=[DictLogger()],
    trackers=[Tracker("val_rmse")]
)

model_trainer.run(1000)

model_trainer.trackers[0].best_metric()
fitted_model = model_trainer.trackers[0].best_model()

eqx.tree_serialise_leaves("created_model.eqx", fitted_model)