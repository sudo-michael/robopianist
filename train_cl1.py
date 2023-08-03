# originally from https://github.com/RolandZhu/robopianist

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from sbx import DroQ
from train import make_env_cl


if __name__ == '__main__':
    # song = 'FantaisieImpromptu'
    song1 = 'TwinkleTwinkleRousseau'
    # song1 = 'CMajorScaleOneHand'
    song2 = 'CMajorScaleTwoHands'
    # song3 = 'LaCampanella'
    # song3 = 'JeTeVeux'
    song3 = 'TheEntertainer'
    # song3 = 'TwinkleTwinkleRousseau'  # hard: song3> song2 > song1
    timestep1 = 4e5  # 2e5+2e5+6e5 = 1e6
    timestep2 = 6e5
    timestep3 = 1e6   #  song3 = 'TwinkleTwinkleRousseau'  # hard: song3> song2 > song1
    # timestep1 = 1e3  # 2e5+2e5+6e5 = 1e6
    # timestep2 = 1e3
    # timestep3 = 1e3

    env1 = make_env_cl(song1, 0, False, timestep=timestep1)()

    # setup logger1
    tmp_path1 = f"./logs/cl/cl1/{song1}{timestep1}"
    tmp_path2 = f"./logs/cl/cl1/{song2}{timestep2}"
    tmp_path3 = f"./logs/cl/cl1/{song3}{timestep3}"

    new_logger1 = configure(tmp_path1, ["stdout", "csv", "tensorboard"])
    new_logger2 = configure(tmp_path2, ["stdout", "csv", "tensorboard"])
    new_logger3 = configure(tmp_path3, ["stdout", "csv", "tensorboard"])


    # setup model1
    model1 = DroQ("MlpPolicy", env1, verbose=1)
    model1.set_logger(new_logger1)
    model1.learn(total_timesteps=timestep1, progress_bar=True)
    model1.save(f'models/cl/cl1/m1_{song1}_{timestep1}')
    print("The training of model1 is finished.")

    del model1

    env2 = make_env_cl(song2, 0, False, timestep=timestep2)()
    model2 = DroQ.load(f'models/cl/cl1/m1_{song1}_{timestep1}', env=env2)  # the load environment is not sure
    print("The trained model1 has been loaded!")
    model2.set_logger(new_logger2)
    model2.learn(total_timesteps=timestep2, progress_bar=True)
    model2.save(f'models/cl/cl1/m2_{song2}_{timestep2}')
    print("The training of model2 is finished.")

    del model2

    env3 = make_env_cl(song3, 0, False, timestep=timestep3)()
    model3 = DroQ.load(f'models/cl/cl1/m2_{song2}_{timestep2}', env=env3)  # the load environment is not sure
    print("The trained model2 has been loaded!")
    model3.set_logger(new_logger3)
    model3.learn(total_timesteps=timestep3, progress_bar=True)
    model3.save(f'models/cl/cl1/m3_{song3}_{timestep3}')
    print("The training of model3 is finished.")