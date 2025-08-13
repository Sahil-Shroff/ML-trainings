import gymnasium
import numpy as np
import matplotlib.pyplot as plt

env = gymnasium.make('FrozenLake-v1', is_slippery=False)
num_states = env.observation_space.n
num_actions = env.action_space.n

V = np.zeros(num_states)
policy = np.random.randint(0, num_actions, size=num_states)
alpha = .1
gamma = .99
epsilon = .1 # exploration rate
episodes = 1000

plt.ion()
fig, ax = plt.subplots()
ax.set_title('TD Learning with Policy Improvement')
ax.set_xlabel('State')
ax.set_ylabel('Value')
bar_plot = ax.bar(range(num_states), V)

def choose_action(state):
    """
    Choose an action based on the current policy with epsilon-greedy exploration 
    """

    if np.random.rand() < epsilon:
        return env.action_space.sample()
    else:
        return policy[state]

def update_policy():
    """
    Policy improvement step:
    Update the policy by selecting the action action that maximizes the expected value.
    """
    for state in range(num_states):
        action_values = []
        for action in range(action_values):
            env.reset()
            next_state, reward, done, _, _ = env.step(action)
            action_values.append(reward + gamma * V[next_state])
        policy[state] = np.argmax(action_values)

for episode in range(episodes):
    state = env.reset()[0]
    done = False
    while not done:
        action = choose_action(state)

        next_state, reward, done, _, _ = env.step(action)
        V[state] += alpha * (reward + gamma * V[next_state] - V[state])
        state = next_state

    update_policy()

    for bar, value in zip(bar_plot, V):
        bar.set_height(value)

    fig.canvas.draw()
    fig.canvas.flush_events()

print(V)
plt.ioff()
plt.show()
env.close()

print("Final Value Function:", V)
print("Final Policy (0: Left, 1: Down, 2: Right, 3: Up):", policy)
