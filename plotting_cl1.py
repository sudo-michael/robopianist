import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


song1 = 'CMajorScaleOneHand'
song2 = 'CMajorScaleTwoHands'

# song3 = 'TheEntertainer (CL)'
# song = 'TheEntertainer'

song3 = 'JeTeVeux (CL)'
song = 'JeTeVeux'


# delete the first row of generated csv files
# song: The Entertainer
# f = 'results/cl/cl1(TheEntertainer)/TheEntertainer_2000000.0.monitor.csv'
# f1 = 'results/cl/cl1(TheEntertainer)/CMajorScaleOneHand_400000.0.monitor.csv'
# f2 = 'results/cl/cl1(TheEntertainer)/CMajorScaleTwoHands_600000.0.monitor.csv'
# f3 = 'results/cl/cl1(TheEntertainer)/TheEntertainer_1000000.0.monitor.csv'

# song: JeTeVeux
f = 'JeTeVeux_2000000.0.monitor.csv'
f1 = 'results/cl/cl(4e5_6e5_1e6)/CMajorScaleOneHand_400000.0.monitor.csv'
f2 = 'results/cl/cl(4e5_6e5_1e6)/CMajorScaleTwoHands_600000.0.monitor.csv'
f3 = 'results/cl/cl(4e5_6e5_1e6)/JeTeVeux_1000000.0.monitor.csv'

df = pd.read_csv(f)
data = df[['f1']]
num = len(data['f1'])

df1 = pd.read_csv(f1)
data1 = df1[['f1']]
num1 = len(data1['f1'])
# print(num1)

df2 = pd.read_csv(f2)
data2 = df2[['f1']]
num2 = len(data2['f1'])

df3 = pd.read_csv(f3)
data3 = df3[['f1']]
num3 = len(data3['f1'])

x = np.linspace(start=0, stop=2e6, num=num)
x1 = np.linspace(start=0, stop=4e5, num=num1)
x2 = np.linspace(start=4e5, stop=1e6, num=num2)
x3 = np.linspace(start=1e6, stop=2e6, num=num3)
# x2 = 

plt.figure()
plt.title('Performances of CL and normal training')
plt.plot(x, data['f1'], color='blue', linewidth=1.0, label=f'{song}')
plt.plot(x1, data1['f1'], color='orange', linewidth=1.0, label=f'{song1}')
plt.plot(x2, data2['f1'], color='purple', linewidth=1.0, label=f'{song2}')
plt.plot(x3, data3['f1'], color='red', linewidth=1.0, label=f'{song3}')
plt.xlabel('timestep')
plt.ylabel('F1 scores')
plt.legend()
plt.grid()
plt.show()


