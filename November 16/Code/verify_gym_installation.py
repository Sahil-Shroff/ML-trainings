import gymnasium

env = gymnasium.make('CartPole-v1')
state = env.reset()
print('Environment initialized successfully!')
env.close()