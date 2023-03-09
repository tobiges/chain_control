#!/usr/bin/env python
# coding: utf-8

# In[1]:


from cc.model import LinearModel, LinearModelOptions, NonlinearModelOptions, NonlinearModel
from cc.model.eval_model import eval_model
from cc.collect import collect_sample
from cc.env import make_env, ModelBasedEnv
from cc.train import train_model, TrainingOptionsModel
from cc import save, load 
import jax.random as jrand
import optax  


# In[2]:


time_limit = 10.0
control_timestep = 0.01

env = make_env("two_segments_v1", time_limit=time_limit, control_timestep=control_timestep, random=1)


# In[4]:


train_sample = collect_sample(
    env,
    seeds_gp=[0,1,2,4,5,6,7,8,9,10,11,12,13,14],
    seeds_cos=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
)

test_sample = collect_sample(
    env, 
    seeds_gp=[15, 16, 17, 18, 19],
    seeds_cos=[2.5, 5.0, 7.5, 10.0, 12.5] # really shouldn't be called seeds, rather frequency
)


# In[5]:


options = NonlinearModelOptions(
    12, 1, 1, # state-size, input-size, output-size
    "EE", # integrate-method
    jrand.PRNGKey(1,), # seed for parameter init
    depth_f=2, # number of layers 
    width_f=25, # width of layers
    depth_g=0,
)


# In[6]:


model = NonlinearModel(options)


# In[7]:


import numpy as np


action = np.array([0.2])
# this returns a new model with an updated internal state
# and of course the actual prediction of the observation
new_model, predicted_obs = model(action)


# In[8]:


predicted_obs


# In[9]:


type(new_model)


# In[10]:


training_options = TrainingOptionsModel(
    optax.adam(3e-3), 0.05, 1000, 1, True 
)

# requires ~25 seconds on my PC
# achieves a Test-Loss of ~4.2 on v1
model, losses = train_model(model, train_sample, training_options=training_options, test_sample=test_sample)


# In[11]:


# small little helper function
predicted_observation, test_rmse = eval_model(model, test_sample)


# In[12]:


predicted_observation["xpos_of_segment_end"].shape


# In[13]:


test_rmse


# Finally, you can also replace the `Mujoco` physics component in your environment with your model. 
# 
# This creates a new environment that looks exactly the same from outside.

# In[14]:


env_model = ModelBasedEnv(env, model, time_limit=time_limit, control_timestep=control_timestep)


# In[15]:


env_model.step([0.2])


# In[16]:


# save model 
save(model, "model_for_two_segments_v1.pkl")


# In[17]:


load("model_for_two_segments_v1.pkl")


# In[ ]:




