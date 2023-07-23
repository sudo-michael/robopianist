from IPython.display import HTML
from base64 import b64encode
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

# Reference: https://stackoverflow.com/a/60986234.
def play_video(filename: str):
    mp4 = open(filename, "rb").read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()

    return HTML(
        """
  <video controls>
        <source src="%s" type="video/mp4">
  </video>
  """
        % data_url
    )


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

def trans(dstate):
    # dstate: the state in the form of dictionary
    state = np.concatenate([v for k,v in sorted(dstate.items())], 0)
    return state