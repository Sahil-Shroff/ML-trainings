import matplotlib.pyplot as plt
import numpy as np

np.random.seed(10)

data1 = np.random.normal(100, 10, 200)
data2 = np.random.normal(90, 20, 200)
data3 = np.random.normal(80, 15, 200)

# print(data1)
# print(data2)
# print(data3)

plt.boxplot([data1, data2, data3], label=['Data 1', 'Data 2', 'Data 3'])
plt.show()