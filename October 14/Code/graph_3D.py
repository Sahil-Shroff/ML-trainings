import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

'''
z=np.linspace(0, 1, 100)
x = z*np.tan(25*z)
y = z*np.cos(25*z)

ax.plot(x, y, z, label='3D Line')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.legend()
'''

'''
x = np.linspace(-5 , 5, 100)
y = np.linspace(-5 , 5, 100)
x, y = np.meshgrid(x, y)
z= np.sin(np.sqrt(x**2 + y**2))

surf = ax.plot_surface(x, y, z, cmap='Spectral')
fig.colorbar(surf)
'''

x=np.random.standard_normal(100)
y=np.random.standard_normal(100)
z=np.random.standard_normal(100)

ax.scatter(x, y, z, c='r', alpha=.5, marker='o')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()