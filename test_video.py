from train import make_env
from sbx import DroQ


if __name__ == '__main__':
    ################ PPO hyperparameters ################
    # lr = 3e-4    # learning rate
    # # n_steps = 1e6 # The number of steps to run for each environment per update
    # batch_size = 256
    # # n_epochs = # 
    # gamma = 0.99
    # song = 'TwinkleTwinkleRousseau'
    # env = make_env(song, 0, False)()
    # model = PPO("MultiInputPolicy", env, learning_rate=lr, batch_size=batch_size)
    # model = PPO.load('models/ppo/twinkle_test_2')

    # song = 'FantaisieImpromptu'  
    # song = 'TwinkleTwinkleRousseau'
    song = 'LaCampanella'
    # song = 'CMajorScaleTwoHands'
    timestep = 1e6
    env = make_env(song, 0, True, timestep=timestep)()
    obs = env.reset()
    # print(f"The shape of obs is {obs.shape}")
    print(f"The observation shape is {env.observation_space}")
    print(f"The action space is {env.action_space}")

    # model = DroQ.load('models/sbx/droq/FantaisieImpromptu', env=env)# models/sbx/droq/FantaisieImpromptu.zip f'models/sbx/droq/{song}'
    # model = DroQ.load('models/sbx/droq/FantaisieImpromptu_10k', env=env)# models/sbx/droq/FantaisieImpromptu.zip f'models/sbx/droq/{song}'
    model = DroQ.load(f'models/sbx/droq/{song}_{timestep}', env=env)

    print("The model is loaded!")

    obs, _ = env.reset()
    print(f"The initial obs is {obs}")
    rewards = 0.0
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        # print(done)
        rewards += reward
        # env.render("human")
    print(f"The show is over.")
    print(f"The final rewards is {rewards}.")
