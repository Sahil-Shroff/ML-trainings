import gymnasium
import numpy as np
import matplotlib.pyplot as plt

env = gymnasium.make('FrozenLake-v1', is_slippery=False)
num_states = env.observation_space.n

V = np.zeros(num_states)
alpha = .1
gamma = .99
episodes = 1000

plt.ion()
fig, ax = plt.subplots()
ax.set_title('TD Learning: State Values')
ax.set_xlabel('State')
ax.set_ylabel('Value')
bar_plot = ax.bar(range(num_states), V)

for episode in range(episodes):
    state = env.reset()[0]
    done = False
    while not done:
        action = env.action_space.sample()
        next_state, reward, done, _, _ = env.step(action)
        V[state] += alpha * (reward + gamma * V[next_state] - V[state])
        state = next_state

    for bar, value in zip(bar_plot, V):
        bar.set_height(value)
    plt.pause(.01)

print(V)
plt.ioff()
plt.show()