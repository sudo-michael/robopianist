# originally from https://github.com/RolandZhu/robopianist

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from sbx import DroQ
from train import make_env


if __name__ == '__main__':
    # song = 'FantaisieImpromptu'
    # song = 'LaCampanella'
    # song = 'CMajorScaleTwoHands'
    song = 'TwinkleTwinkleRousseau'
    timestep = 1e6
    env = make_env(song, 0, False)()

    # setup logger
    tmp_path = f"./logs/sbx/droq/{song}{timestep}"
    new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

    # setup model
    model = DroQ("MlpPolicy", env, verbose=1)
    model.set_logger(new_logger)
    model.learn(total_timesteps=timestep, progress_bar=True)
    model.save(f'models/sbx/droq/{song}_{timestep}')

    del model

    # env2 = make_env(song, 0, True)()
    model = DroQ.load(f'models/sbx/droq/{song}_{timestep}', env=env)
    print("The model is loaded!")

    # obs = env2.reset()
    # print(obs)
    # action, _states = model.predict(obs)
    # rewards = 0.0
    # done = False
    # while not done:
    #     action, _states = model.predict(obs)
    #     obs, reward, done, truncated, info = env2.step(action)
    #     rewards += reward
    #     # env2.render("human")

    # print(f"The final rewards is {rewards}.")

