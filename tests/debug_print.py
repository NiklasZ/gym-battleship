import gym
import gym_battleship

env = gym.make('Battleship-v0')
env.reset()

for i in range(10):
    action = env.action_space.sample()
    print(f'\nFiring at {action}')
    env.step(action)
    env.render()

env.render_board_generated()
env.render()
