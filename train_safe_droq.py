# originally from https://github.com/RolandZhu/robopianist

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from sbx import DroQLag
from train import make_safe_env


if __name__ == '__main__':
    song = 'TwinkeTwinkleRousseau'
    timestep = 1e6
    env = make_safe_env(song, 0, False, timestep=timestep)()

    # setup logger
    tmp_path = f"./logs/sbx/droqlag/{song}{timestep}"
    new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

    # setup model
    model = DroQLag("MlpPolicy", env, verbose=1)
    model.set_logger(new_logger)
    model.learn(total_timesteps=timestep, progress_bar=True)
    model.save(f'models/sbx/droqlag/{song}_{timestep}')
