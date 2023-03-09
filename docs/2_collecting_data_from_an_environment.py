#!/usr/bin/env python
# coding: utf-8

# # Collecting data from an environment

# In[31]:


from cc.collect import (
    collect,
    collect_reference_source,
    collect_sample,
    concat_iterators
)
from cc import save, load 
from cc.env import make_env
from cc.utils import generate_ts
from cc.env.wrappers import AddReferenceObservationWrapper
from cc.env.wrappers.add_source import default_reward_fn
import matplotlib.pyplot as plt 


# In[5]:


time_limit=5.0 # s
control_timestep=0.01 # s 

env = make_env(
    "two_segments_v1", 
    random=1, # the initial condition might be randomized; fixes the seed 
    control_timestep=control_timestep, # the control rate, here 100 Hz 
    time_limit=time_limit, # the upper time limit; after 5 seconds the environment is done
    delay=0 # whether or not the action is delayed before the environment sees it 
    )


# In[8]:


ts = generate_ts(time_limit, control_timestep)


# ##### Therefore the length of an episode trajectory will be

# In[9]:


N = len(ts)+1 # +1 for the initial state (no action performed yet)
N


# #### And we apply inputs at those timesteps

# In[10]:


ts[:20]


# #### The input to the system and observation from the system is of the form

# In[11]:


env.action_spec()


# In[12]:


env.observation_spec()


# In[13]:


env.step([0.1])


# In[14]:


env.reset()


# ##### Let's collect some data from the environment

# In[16]:


train_sample = collect_sample(
    env, 
    seeds_gp=[0,1,2], # this uses a gaussian process to draw some fixed action/input trajectories 
                        # that we then can apply to the system and record the output
    seeds_cos=[2,4] # this uses a cosine-wave with a frequency of 2 and 4; really this shouldn't 
                        # be called seed 
    )


# In[25]:


type(train_sample)


# In[18]:


train_sample.bs


# In[20]:


train_sample.action.shape


# In[21]:


train_sample.obs["xpos_of_segment_end"].shape


# In[24]:


for idx in range(train_sample.bs):
    plt.plot(train_sample.action[idx])
plt.ylabel("Motor input")


# In[23]:


for idx in range(train_sample.bs):
    plt.plot(train_sample.obs["xpos_of_segment_end"][idx])
plt.ylabel("X-Position")


# In[29]:


train_sample.rew[0,:10]


# Notice how there is no reward, how could there be? 
# 
# We have to add 
# - first, create some reference output that we want to track
# - second, add the reference to the environment and specify a reward-function

# Let's now create an arbitrary smooth input-trajectory, record the observation of that input as reference and store it 

# In[30]:


source = collect_reference_source(env, 
    seeds = [0,17], # this should really be called seeds_gp 
    constant_after=False, # if we enable this flag the reference will become constant after 3 seconds 
    constant_after_T=3.0 # unused 
)


# In[32]:


env_w_rew = AddReferenceObservationWrapper(env, 
    source=source,
    reward_fn=default_reward_fn 
)


# Now, the observation is the reference and the actual observation

# In[34]:


env_w_rew.observation_spec()


# And the environment now has some reward

# In[41]:


env_w_rew.step([0.1]).reward


# In[42]:


env_w_rew.step([0.1]).reward


# Remember, how our source contains *2* trajectories. So, which one are we currently "seeing" in this environment in the observation

# In[43]:


source._i_actor


# In[45]:


env_w_rew.reset()
env_w_rew.step([0.1]).observation["ref"]


# In[46]:


# let's change the reference to the second one 
source.change_reference_of_actor(1)
source._i_actor


# In[47]:


# now it is different
env_w_rew.reset()
env_w_rew.step([0.1]).observation["ref"]


# ### Now we have all the components to 
# - get Training/Testing-data for the training of a *model* using `collect_sample`
# - get suible Reference Trajectories using `collect_reference_source` and add this reference to the environment

# ##### Finally, you can also save/load `source` and `train_sample` using the functions `save` and `load`

# In[28]:


save(train_sample, "train_sample.pkl")
load("train_sample.pkl")


# In[ ]:




