import matplotlib.pyplot as plt
import optax

from cc.collect import collect_reference_source, collect_exhaust_source, collect
from cc import save, load
from cc.controller import create_pi_controller, LinearController, LinearControllerOptions
from cc.env.model_based_env import ModelBasedEnv
from cc.env.wrappers import AddReferenceObservationWrapper, RecordVideoWrapper
from cc.env import make_env
from cc.train import TrainingOptionsController, train_controller
from cc.utils.utils import generate_ts, extract_timelimit_timestep_from_env

time_limit = 10.0
control_timestep = 0.01
ts = generate_ts(time_limit, control_timestep)

# Environment creation
env = make_env("two_segments_v1", random=1, time_limit=time_limit,
               control_timestep=control_timestep)

# Model Loading
model = load("../docs/model_for_two_segments_v1.pkl")
source = collect_reference_source(
    env, seeds=[20], constant_after=True, constant_after_T=3.0)

# Training
p_gain = 0.01
i_gain = 0.0

options = create_pi_controller(p_gain, i_gain, delta_t=control_timestep)
controller = LinearController(options)

training_options = TrainingOptionsController(
    optax.adam(3e-3), 0.0, 500, 1, models=[model]
)

controller, losses = train_controller(controller, source, training_options)
time_limit, control_timestep, ts = extract_timelimit_timestep_from_env(env)

real_env_w_source = ModelBasedEnv(env, model, time_limit=time_limit, control_timestep=control_timestep) # <--- collect 
real_env_w_source = AddReferenceObservationWrapper(real_env_w_source, source)
real_env_iterator = collect(env=real_env_w_source, controller=controller, ts=ts)
real_env_sample = next(real_env_iterator)


print("preparing plots")

plt.plot(real_env_sample.obs["obs"]
         ["xpos_of_segment_end"][0], label="observation environment")
plt.plot(real_env_sample.obs["ref"]
         ["xpos_of_segment_end"][0], label="reference environment")

plt.legend()
plt.show()