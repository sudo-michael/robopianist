# originally from https://github.com/RolandZhu/robopianist

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from sbx import DroQ
from train import make_env



if __name__ == '__main__':
    # song = 'FantaisieImpromptu'
    song = 'TwinkleTwinkleRousseau'
    env = make_env(song, 0, False)()
    tmp_path = f"./logs/sbx/droq/{song}"
    # setup logger
    new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    # setup model
    model = DroQ("MultiInputPolicy", env, verbose=1)
    model.set_logger(new_logger)
    model.learn(total_timesteps=10_000, progress_bar=True)
    model.save(f'models/sbx/droq/{song}_test')

    del model

    model = DroQ.load(f'models/sbx/droq/{song}_test', env=env)
    print("The model is loaded!")

