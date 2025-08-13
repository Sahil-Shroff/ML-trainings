import matplotlib.pyplot as plt
import numpy as np

Xpoints=np.array([1, 2, 3, 4, 6])
Ypoints=np.array([5, 8, 3, 7, 8])
plt.plot(Xpoints, Ypoints, 'h', ms='20', mfc='#4CAF50', linestyle=':', color='r')

plt.xlabel('Heart Rate')
plt.title('Report', loc='left')
plt.grid()
# plt.scatter(Xpoints, Ypoints)
plt.bar(Xpoints, Ypoints)
plt.show()