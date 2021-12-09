from envs.classic_control.cartpole  import CartPoleEnv
from envs.classic_control.pendulum import PendulumEnv
from envs.classic_control.acrobot import AcrobotEnv
from envs.classic_control.continuous_mountain_car import Continuous_MountainCarEnv
from envs.classic_control.mountain_car import MountainCarEnv
from envs.classic_control.pvtol import PvtolEnv
from envs.classic_control.quadrotor import QuadrotorEnv


env = PvtolEnv()
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