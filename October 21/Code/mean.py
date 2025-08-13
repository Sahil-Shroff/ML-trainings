import numpy as np
import matplotlib.pyplot as plt
# from sklearn.metrics import mean_absolute_error

datapoints = np.random.rand(10)

mean = datapoints.mean()
var = datapoints.var()
std = datapoints.std()

print(mean)
print(std)

plt.figure(figsize=(10, 6))

plt.scatter(range(len(datapoints)), datapoints, color='black', label='Actual Values', linewidths=.2)
# plt.plot(range(len(datapoints)), [mean] * len(datapoints), color='blue', label='Mean', linewidth=2)

plt.axhline(y=mean, color='blue', label='mean')
plt.axhline(y=mean+var, color='orange', label='mean + var')
plt.axhline(y=mean-var, color='orange', label='mean - var')
plt.axhline(y=mean+std, color='pink', label='mean + std')
plt.axhline(y=mean-std, color='pink', label='mean - std')

plt.legend(loc='upper right')
plt.show()