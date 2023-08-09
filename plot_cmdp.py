# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# song = 'TwinkeTwinkleRousseau'
song = 'TheEntertainer'

# delete the first row of generated csv files
f = f"./results/{song}_DroqLag_1_1000000.monitor.csv"
f1 = f"./results/{song}_DroqLag_2_1000000.monitor.csv"
f2 = f"./results/{song}_DroqLag_2_1000000.monitor.csv"

# %%
df = pd.read_csv(f)
data = df[['f1']]

df1 = pd.read_csv(f1)
data1 = df1[['f1']]

df2 = pd.read_csv(f2)
data2 = df2[['f1']]

# %%
mean = (data['f1'] + data1['f1'] + data2['f1']) / 3
# %%
std = np.std(np.concatenate((data['f1'], data1['f1'], data2['f1']), axis=-1), axis=-1) / np.sqrt(3)

x = np.linspace(start=0, stop=1e6, num=len(df))
plt.figure()
plt.title('CMDP RoboPianist')
plt.plot(x, mean, color='blue', linewidth=1.0, label=f'DroqLag')
plt.fill_between(x, mean - std, mean + std, alpha=0.2, color='blue')
plt.xlabel('timestep')
plt.ylabel('F1 Score')
plt.legend()
plt.grid()
# %%
df = pd.read_csv(f)
df1 = pd.read_csv(f1)


x = np.linspace(start=0, stop=1e6, num=len(df))
# x3 = np.linspace(start=1e6, stop=2e6, num=num3)
# x2 = 

plt.figure()
plt.title('CMDP RoboPianist')
plt.plot(x, df['rollout/ep_rew_mean'], color='blue', linewidth=1.0, label=f'DroqLag')
plt.plot(x, df1['rollout/ep_rew_mean'], color='red', linewidth=1.0, label=f'Droq')
plt.xlabel('timestep')
plt.ylabel('reward')
plt.legend()
plt.grid()
plt.savefig("cmdp_data.png")



plt.figure()
plt.title('CMDP RoboPianist')
plt.plot(x, df['rollout/ep_cost_mean'], color='blue', linewidth=1.0, label=f'DroqLag')
plt.xlabel('timestep')
plt.ylabel('cost')
plt.legend()
plt.grid()
plt.savefig("cmdp_cost.png")


