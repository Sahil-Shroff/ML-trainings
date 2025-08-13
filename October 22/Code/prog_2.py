import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation 
from IPython.display import HTML

fig, ax = plt.subplots()

x = np.linspace(0, 2 *np.pi, 100)
line, = ax.plot(x, np.sin(x))

def init():
    line.set_ydata(np.ma.array(x, mask=True))
    return line,

def update(frame):
    line.set_ydata(np.sin(x + frame / 10.0))
    return line,

ani = animation.FuncAnimation(fig, update, frames=np.arange(0, 100), init_func=init, blit=True)

HTML(ani.to_jshtml())

plt.show()
