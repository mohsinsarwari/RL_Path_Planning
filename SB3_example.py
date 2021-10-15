import gym
import numpy as np

from stable_baselines3.sac import MlpPolicy
from stable_baselines3 import SAC

env = gym.make('Pendulum-v1')

model = SAC(MlpPolicy, env, verbose=1, tensorboard_log="SB3_SAC_example")
model.learn(total_timesteps=1000000, log_interval=10, tb_log_name="example_log")
model.save("sac_pendulum")

del model # remove to demonstrate saving and loading

model = SAC.load("sac_pendulum")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

env.close()


