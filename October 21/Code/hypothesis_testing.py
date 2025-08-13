import numpy as np
from scipy.stats import chi2_contingency

observed_frequency = np.array([[25, 15, 10], [30, 20, 15], [20, 25, 15]])


res = chi2_contingency(observed_frequency)
print(res)