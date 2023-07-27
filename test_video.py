# originally from https://github.com/RolandZhu/robopianist
import os
import sys

import numpy as np
from robopianist.suite.tasks import self_actuated_piano
from robopianist.suite.tasks import piano_with_shadow_hands
from dm_env_wrappers import CanonicalSpecWrapper, ConcatObservationWrapper
from robopianist.wrappers import PianoSoundVideoWrapper, MidiEvaluationWrapper
from robopianist import music
from mujoco_utils import composer_utils
import dm_env
from dmcgym import DMCGYM
import shimmy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3 import PPO
from sbx import DroQ



def make_env(song: str = 'TwinkleTwinkleRousseau', 
             seed: int = 0,
             sound: bool = False,
             log_dir='./logs'):
    """
    Utility function for multiprocessed env.
    :param song: the name of the song
    :param seed: the inital seed for RNG
    """
    def _init():
        task = piano_with_shadow_hands.PianoWithShadowHands(
            change_color_on_activation=True,
            midi=music.load(song),
            trim_silence=True,
            control_timestep=0.05,
            gravity_compensation=True,
            primitive_fingertip_collisions=False,
            reduced_action_space=False,
            n_steps_lookahead=10,
            disable_fingering_reward=False,
            disable_forearm_reward=False,
            disable_colorization=False,
            disable_hand_collisions=False,
            attachment_yaw=0.0,
        )

        env = composer_utils.Environment(
            task=task, strip_singleton_obs_buffer_dim=True, recompile_physics=False
        )

        env = ConcatObservationWrapper(env)
        env = CanonicalSpecWrapper(env)
        if sound:
            env = PianoSoundVideoWrapper(
            env,
            record_every=1,
            camera_id=None, # "piano/back",
            record_dir="./videos",
            )
        env = MidiEvaluationWrapper(env)
        env = DMCGYM(env)
        env = shimmy.GymV21CompatibilityV0(env=env)
        env = Monitor(env, filename=f'{song}', info_keywords=('precision', 'recall', 'f1', 'sustain_precision', 'sustain_recall', 'sustain_f1'))

        return env

    # set_random_seed(seed)

    return _init


if __name__ == '__main__':
    ################ PPO hyperparameters ################
    lr = 3e-4    # learning rate
    # n_steps = 1e6 # The number of steps to run for each environment per update
    batch_size = 256
    # n_epochs = # 
    gamma = 0.99
    
    # song = 'TwinkleTwinkleRousseau'
    # env = make_env(song, 0, False)()
    # model = PPO("MultiInputPolicy", env, learning_rate=lr, batch_size=batch_size)
    # model = PPO.load('models/ppo/twinkle_test_2')

    song = 'FantaisieImpromptu'
    env = make_env(song, 0, False)()
    # tmp_path = f"./logs/sbx/droq/{song}"
    model = DroQ("MlpPolicy", env, verbose=1)
    model = DroQ.load('models/sbx/droq/FantaisieImpromptu')# models/sbx/droq/FantaisieImpromptu.zip f'models/sbx/droq/{song}'

    print("Model is loaded!")

    obs = env.reset()
    print(obs)
    action, _states = model.predict(obs)
    rewards = 0.0
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        rewards += reward
        env.render("human")

    print(f"The final rewards is {rewards}.")
