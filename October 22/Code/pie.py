import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation 
from IPython.display import HTML

fig, ax = plt.subplots()

data = [20, 30, 10, 40]
wedges, texts = ax.pie(data)

def init():
    ax.clear()
    wedges, _ = ax.pie(data)
    return wedges

def update(frame):
    new_data = [20 + np.sin(frame / 10.0) * 10, 30 + np.cos(frame / 10.0) * 10, 10 , 40]
    ax.clear()
    wedges, _ = ax.pie(new_data)
    return wedges

ani = animation.FuncAnimation(fig, update, frames=np.arange(0, 100), init_func=init, blit=True)

HTML(ani.to_jshtml())

plt.show()
