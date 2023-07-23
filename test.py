# originally from https://github.com/RolandZhu/robopianist
import os
import sys

import numpy as np
from robopianist.suite.tasks import self_actuated_piano
from robopianist.suite.tasks import piano_with_shadow_hands
from dm_env_wrappers import CanonicalSpecWrapper
from robopianist.wrappers import PianoSoundVideoWrapper
from robopianist import music
from mujoco_utils import composer_utils
import dm_env
from dmcgym import DMCGYM
from utilities import play_video

def make_env(song: str = 'TwinkleTwinkleRousseau', 
             seed: int = 0,
             sound: bool = False,):
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

        if sound:
            env = PianoSoundVideoWrapper(
                env,
                record_every=1,
                camera_id=None, # "piano/back",
                record_dir="./videos",
            )
        env = CanonicalSpecWrapper(env)

        env = DMCGYM(env)

        return env

    # set_random_seed(seed)

    return _init

class Policy:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._idx = 0
        self._actions = np.load("twinkle_twinkle_actions.npy")

    def __call__(self, timestep: dm_env.TimeStep) -> np.ndarray:
        del timestep  # Unused.
        actions = self._actions[self._idx]
        self._idx += 1
        return actions


if __name__ == '__main__':
    env = make_env('TwinkleTwinkleRousseau', 0, False)()
    # print(env.action_space)  # 45
    obs_dim = 0
    for k, v in env.observation_space.items():
        obs_dim += int(np.prod(v.shape))
    # print(f"The observation dimension is {obs_dim}.")  # 1136

    state = env.reset()
    policy = Policy()

    done = False
    while not done:
        action = policy(state)
        state, reward, done, _ = env.step(action)

    play_video('test')

    

