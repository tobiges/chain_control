#!/usr/bin/env python
# coding: utf-8

# In[39]:


from cc.collect import collect_reference_source, collect_exhaust_source, collect 
from cc import save, load 
from cc.controller import create_pi_controller, LinearController, LinearControllerOptions
from cc.env.wrappers import AddReferenceObservationWrapper, RecordVideoWrapper
import matplotlib.pyplot as plt 
from cc.env import make_env
from cc.train import TrainingOptionsController, train_controller
import optax 


# In[24]:


from cc.utils.utils import generate_ts


time_limit = 10.0
control_timestep = 0.01 
ts = generate_ts(time_limit, control_timestep)

env = make_env("two_segments_v1", random=1, time_limit=time_limit, control_timestep=control_timestep)


# In[3]:


# we trained this model in the notebook #3
model = load("model_for_two_segments_v1.pkl")


# In[4]:


# create a reference source with two *feasible* (=physically possible) trajectories
source = collect_reference_source(env, seeds=[20,21], constant_after=True, constant_after_T=3.0)


# In[16]:


p_gain = 0.01
i_gain = 0.0 

options = create_pi_controller(p_gain, i_gain, delta_t=control_timestep)


# In[17]:


controller = LinearController(options)


# In[18]:


controller


# In[19]:


training_options = TrainingOptionsController(
    optax.adam(3e-3), 0.0, 500, 1, models=[model]
)


# In[20]:


controller, losses = train_controller(controller, source, training_options)


# In[22]:


# final gain values
controller.rhs.params.C()


# ### So how well does our controller now perform in the actual environment? :D 

# In[26]:


env_w_source = AddReferenceObservationWrapper(env, source)


# In[28]:


iterator = collect(env=env_w_source, controller=controller, ts=ts)
replay_sample = next(iterator)


# In[35]:


replay_sample.obs


# In[36]:


plt.plot(replay_sample.obs["obs"]["xpos_of_segment_end"][0], label="observation")
plt.plot(replay_sample.obs["ref"]["xpos_of_segment_end"][0], label="reference")
plt.legend()


# In[37]:


source.change_reference_of_actor(1)
iterator = collect(env=env_w_source, controller=controller, ts=ts)
replay_sample = next(iterator)


# In[38]:


plt.plot(replay_sample.obs["obs"]["xpos_of_segment_end"][0], label="observation")
plt.plot(replay_sample.obs["ref"]["xpos_of_segment_end"][0], label="reference")
plt.legend()


# Want a video?

# In[41]:


# This Functionality probably only works on Mac or Linux

source.change_reference_of_actor(0)

# cleanup_imgs is kind of dangerous, check out the source code before you run this please
env_w_video = RecordVideoWrapper(env_w_source, width=1280, height=720, cleanup_imgs=False)

iterator = collect(env=env_w_video, controller=controller, ts=ts)


# If you are interested in Performance of the controller in the model, then remember you can always convert a model to an environment using `ModelBasedEnv`.
# 
# You might also want to check out the utility function `cc.collect.collect_exhaust_source` which does all that work for you!

# 
