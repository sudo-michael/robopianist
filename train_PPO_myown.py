from IPython.display import HTML
from base64 import b64encode
import numpy as np
import torch
import torch.nn as nn
from robopianist.suite.tasks import self_actuated_piano
from robopianist.suite.tasks import piano_with_shadow_hands
from dm_env_wrappers import CanonicalSpecWrapper
from robopianist.wrappers import PianoSoundVideoWrapper
from robopianist import music
from mujoco_utils import composer_utils
import dm_env
from utilities import play_video, make_env, trans
from PPOmyown import PPO
import os
from datetime import datetime


################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
# device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    device = torch.device('cpu')
    print("Device set to : cpu")
print(f"The PPO controller is set to {device}.")
print("============================================================================================")

####### initialize environment hyperparameters ######
has_continuous_action_space = False  # continuous action space; else discrete

max_ep_len = 1000                   # max timesteps in one episode
max_training_timesteps = int(1e6)   # break training loop if timeteps > max_training_timesteps

print_freq = max_ep_len * 10        # print avg reward in the interval (in num timesteps)
log_freq = max_ep_len * 2           # log avg reward in the interval (in num timesteps)
save_model_freq = int(2e4)          # save model frequency (in num timesteps)

action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)
#####################################################


################ PPO hyperparameters ################
update_timestep = max_ep_len * 4      # update policy every n timesteps
K_epochs = 80               # update policy for K epochs in one PPO update

eps_clip = 0.2          # clip parameter for PPO
gamma = 0.99            # discount factor

lr_actor = 0.0003       # learning rate for actor network
lr_critic = 0.001       # learning rate for critic network

random_seed = 0         # set random seed if required (0 = no random seed)
#####################################################

################ env setup #################
song = 'TwinkleTwinkleRousseau'
env = make_env(song, 0, False)()
action_dim = env.action_space.shape[0]
print(f"The action dimension is {action_dim}.")  # 45
state_dim = 0
for k, v in env.observation_space.items():
    state_dim += int(np.prod(v.shape))
print(f"The observation dimension is {state_dim}.")  # 1136

###################### logging ######################

#### log files for multiple runs are NOT overwritten
log_dir = "PPO_logs"
if not os.path.exists(log_dir):
        os.makedirs(log_dir)

log_dir = log_dir + '/' + song + '/'
if not os.path.exists(log_dir):
        os.makedirs(log_dir)
#### get number of log files in log directory
run_num = 0
current_num_files = next(os.walk(log_dir))[2]
run_num = len(current_num_files)

#### create new log file for each run
log_f_name = log_dir + '/PPO_' + song + "_log_" + str(run_num) + ".csv"

print("current logging run number for " + song + " : ", run_num)
print("logging at : " + log_f_name)
#####################################################
################### checkpointing ###################
run_num_pretrained = 0      #### change this to prevent overwriting weights in same env_name folder

directory = "PPO_models"
if not os.path.exists(directory):
        os.makedirs(directory)

directory = directory + '/' + song + '/'
if not os.path.exists(directory):
        os.makedirs(directory)
#####################################################
################# training procedure ################
# initialize a PPO agent
ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, device, action_std)

# track total training time
start_time = datetime.now().replace(microsecond=0)
print("Started training at (GMT) : ", start_time)

print("============================================================================================")

# logging file
log_f = open(log_f_name,"w+")
log_f.write('episode,timestep,reward\n')

# printing and logging variables
print_running_reward = 0
print_running_episodes = 0

log_running_reward = 0
log_running_episodes = 0

time_step = 0
i_episode = 0

# training loop
while time_step <= max_training_timesteps:

    state = env.reset()
    state = trans(state)
    current_ep_reward = 0

    for t in range(1, max_ep_len+1):

        # select action with policy
        action = ppo_agent.select_action(state)
        state, reward, done, _ = env.step(action)
        state = trans(state)

        # saving reward and is_terminals
        ppo_agent.buffer.rewards.append(reward)
        ppo_agent.buffer.is_terminals.append(done)

        time_step +=1
        current_ep_reward += reward

        # update PPO agent
        if time_step % update_timestep == 0:
            ppo_agent.update()

        # if continuous action space; then decay action std of ouput action distribution
        if has_continuous_action_space and time_step % action_std_decay_freq == 0:
            ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

        # log in logging file
        if time_step % log_freq == 0:

            # log average reward till last episode
            log_avg_reward = log_running_reward / log_running_episodes
            log_avg_reward = round(log_avg_reward, 4)

            log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
            log_f.flush()

            log_running_reward = 0
            log_running_episodes = 0

        # printing average reward
        if time_step % print_freq == 0:

            # print average reward till last episode
            print_avg_reward = print_running_reward / print_running_episodes
            print_avg_reward = round(print_avg_reward, 2)

            print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))

            print_running_reward = 0
            print_running_episodes = 0

        # save model weights
        if time_step % save_model_freq == 0:
            checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(song, random_seed, time_step)
            print("save checkpoint path : " + checkpoint_path)
            print("--------------------------------------------------------------------------------------------")
            print("saving model at : " + checkpoint_path)
            ppo_agent.save(checkpoint_path)
            print("model saved")
            print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
            print("--------------------------------------------------------------------------------------------")

        # break; if the episode is over
        if done:
            break

    print_running_reward += current_ep_reward
    print_running_episodes += 1

    log_running_reward += current_ep_reward
    log_running_episodes += 1

    i_episode += 1

log_f.close()
env.close()

# print total training time
print("============================================================================================")
end_time = datetime.now().replace(microsecond=0)
print("Started training at (GMT) : ", start_time)
print("Finished training at (GMT) : ", end_time)
print("Total training time  : ", end_time - start_time)
print("============================================================================================")