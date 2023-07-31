import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# delete the first row of generated csv files
# file_name = 'results/cl/cl(4e5_6e5_1e6)/JeTeVeux_1000000.0.monitor.csv' # results/cl/cl(4e5_6e5_1e6)/JeTeVeux_1000000.0.monitor.csv
# file_name = 'results/cl/cl(4e5_6e5_1e6)/CMajorScaleOneHand_400000.0.monitor.csv'
file_name = 'results/cl/cl(4e5_6e5_1e6)/CMajorScaleTwoHands_600000.0.monitor.csv'
file_name = 'results/cl/cl(4e5_6e5_1e6)/total.csv'
file_name = 'JeTeVeux_1000000.0.monitor.csv'
file_name = 'JeTeVeux_2000000.0.monitor.csv'

df = pd.read_csv(file_name)
data = df[['f1', 'precision', 'recall']]
# print(len(data['f1']))

x_data = np.linspace(start=0, stop=len(2e6/data['f1']), num=2e6))  # int(len(data['f1'])
print(len(x_data))

plt.figure()
plt.plot(x_data, data['f1'])
plt.xlabel('timestep')
plt.ylabel('f1 scores')
# plt.show()
plt.savefig(f'figures/{file_name}.png')