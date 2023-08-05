# originally from https://github.com/RolandZhu/robopianist

from stable_baselines3.common.logger import configure
from sbx.droq.droqlag import DroQLag
from train import make_safe_env
import argparse

def get_args()
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    return args

if __name__ == '__main__':
    song = 'TwinkleTwinkleRousseau'
    timestep = 1e6
    args = get_args()
    env = make_safe_env(song, args.seed, False, timestep=timestep)()
    

    # setup logger
    tmp_path = f"./logs/sbx/droqlag/{song}{timestep}"
    new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

    # setup model
    model = DroQLag("MlpPolicy", env, verbose=1, seed=args.seed)
    model.set_logger(new_logger)
    model.learn(total_timesteps=timestep, progress_bar=True)
    # model.save(f'models/sbx/droqlag/{song}_{timestep}')
