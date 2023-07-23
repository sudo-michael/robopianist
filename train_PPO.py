from IPython.display import HTML
from base64 import b64encode
import numpy as np
import torch
import torch.nn as nn
from robopianist.suite.tasks import self_actuated_piano
from robopianist.suite.tasks import piano_with_shadow_hands
from dm_env_wrappers import CanonicalSpecWrapper
from robopianist.wrappers import PianoSoundVideoWrapper
from robopianist import music
from mujoco_utils import composer_utils
import dm_env
from utilities import play_video, make_env, trans
import os
from datetime import datetime
from stable_baselines3 import PPO

################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
# device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    device = torch.device('cpu')
    print("Device set to : cpu")
print(f"The PPO controller is set to {device}.")
print("============================================================================================")

################ env setup #################
song = 'TwinkleTwinkleRousseau'
env = make_env(song, 0, False)()
# action_dim = env.action_space.shape[0]
# print(f"The action dimension is {action_dim}.")  # 45
# state_dim = 0
# for k, v in env.observation_space.items():
#     state_dim += int(np.prod(v.shape))
# print(f"The observation dimension is {state_dim}.")  # 1136

################ PPO hyperparameters ################
lr = 3e-4    # learning rate
# n_steps = 1e6 # The number of steps to run for each environment per update
batch_size = 256
# n_epochs = # 
gamma = 0.99

#####################################################
################ model setup #################
model = PPO("MultiInputPolicy", env, learning_rate=lr, batch_size=batch_size)
model.learn(total_timesteps=1e6)
model.save(f"ppo_{song}")

