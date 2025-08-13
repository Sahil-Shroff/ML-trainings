import numpy as np
from scipy.stats import f_oneway

observed_frequency = np.array([[25, 15, 10], [30, 20, 15], [20, 25, 15]])

res = f_oneway(observed_frequency[0], observed_frequency[1], observed_frequency[2])
print(res)