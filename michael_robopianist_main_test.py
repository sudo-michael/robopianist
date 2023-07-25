# originally from https://github.com/RolandZhu/robopianist
import os
import sys

import numpy as np
from robopianist.suite.tasks import self_actuated_piano
from robopianist.suite.tasks import piano_with_shadow_hands
from dm_env_wrappers import CanonicalSpecWrapper
from robopianist.wrappers import PianoSoundVideoWrapper, MidiEvaluationWrapper
from robopianist import music
from mujoco_utils import composer_utils
import dm_env
from dmcgym import DMCGYM
import shimmy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3 import PPO


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

        # if sound:
        #     env = PianoSoundVideoWrapper(
        #         env,
        #         record_every=100,
        #         camera_id=None, # "piano/back",
        #         record_dir="./videos",
        #     )
        env = CanonicalSpecWrapper(env)
        # NOTE: modify DMCGYM to return music metrics
        env = MidiEvaluationWrapper(env)
        env = DMCGYM(env)
        env = shimmy.GymV21CompatibilityV0(env=env)
        env = Monitor(env, filename='test_monitor', info_keywords=('precision', 'recall', 'f1', 'sustain_precision', 'sustain_recall', 'sustain_f1'))

        return env

    # set_random_seed(seed)

    return _init

if __name__ == '__main__':
    env = make_env('TwinkleTwinkleRousseau', 0, False)()


    tmp_path = "./logs/sb3/ppo"
    # set up logger
    new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

    model = PPO("MultiInputPolicy", env, verbose=1)
    model.set_logger(new_logger)
    model.learn(total_timesteps=1_000_000, progress_bar=True)
    model.save('models/ppo/twinkle_test_2')