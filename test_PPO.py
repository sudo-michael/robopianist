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
from stable_baselines3 import PPO
import os
from datetime import datetime



################ PPO hyperparameters ################
lr = 3e-4    # learning rate
# n_steps = 1e6 # The number of steps to run for each environment per update
batch_size = 256
# n_epochs = # 
gamma = 0.99
song = 'TwinkleTwinkleRousseau'
env = make_env(song, 0, True)()

model = PPO("MultiInputPolicy", env, learning_rate=lr, batch_size=batch_size)
model = PPO.load(f"models/sb3/ppo/{song}")
print("Model is loaded!")

obs = env.reset()
rewards = 0.0
done = False
while not done:
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    rewards += reward
    env.render("human")

print(f"The final rewards is {rewards}.")