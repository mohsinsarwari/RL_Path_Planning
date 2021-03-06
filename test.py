from envs.classic_control import *
from envs.mujoco.walker2d import Walker2dEnv


env = Walker2dEnv()
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        action = env.action_space.sample()
        env.render()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()