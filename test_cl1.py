from train import make_env_cl_test
from sbx import DroQ


if __name__ == '__main__':
    ################ PPO hyperparameters ################
    song1 = 'CMajorScaleOneHand'
    song2 = 'CMajorScaleTwoHands'
    # song3 = 'TwinkleTwinkleRousseau'  # hard: song3> song2 > song1
    song3 = 'LaCampanella'
    timestep1 = 2e5  # 2e5+2e5+6e5 = 1e6
    timestep2 = 2e5
    timestep3 = 6e5    # song3 = 'TwinkleTwinkleRousseau'  # hard: song3> song2 > song1
    # timestep1 = 1e3  # 2e5+2e5+6e5 = 1e6
    # timestep2 = 1e3
    # timestep3 = 1e3 
    song = 'TwinkleTwinkleRousseau'
    # song = 'LaCampanella'
    # song = 'CMajorScaleTwoHands'
    timestep = 1e6
    # env = make_env(song3, 0, True, timestep=timestep)()
    env3 = make_env_cl_test(song3, 0, True, timestep=timestep3)()

    # obs = env.reset()
    # # print(f"The shape of obs is {obs.shape}")
    # print(f"The observation shape is {env.observation_space}")
    # print(f"The action space is {env.action_space}")

    # model = DroQ.load('models/sbx/droq/FantaisieImpromptu', env=env)# models/sbx/droq/FantaisieImpromptu.zip f'models/sbx/droq/{song}'
    # model = DroQ.load('models/sbx/droq/FantaisieImpromptu_10k', env=env)# models/sbx/droq/FantaisieImpromptu.zip f'models/sbx/droq/{song}'
    model = DroQ.load(f'models/cl/cl1/m3_{song3}_{timestep3}', env=env3)

    print("The model is loaded!")

    obs, _ = env3.reset()
    print(f"The initial obs is {obs}")
    rewards = 0.0
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, truncated, info = env3.step(action)
        # print(done)
        rewards += reward
        # env.render("human")
    print(f"The show is over.")
    print(f"The final rewards is {rewards}.")
