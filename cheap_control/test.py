from envs.classic_control import ManipulatorEnv

import os
import gym
import csv
import numpy as np
import torch as th
from io import StringIO
from datetime import datetime
import json
import pickle


param_dict = {
    #path info
    'folder': "Quadrotor_Test",
    'description': "Testing out Quadrotor Simulation",
    #shared params
    'dt': 0.05,
    #RL_env parameters
    'total_time': 5,
    'total_timesteps': 50000,
    #model parameters
    'policy_kwarg': dict(activation_fn=th.nn.Tanh),
    'eval_freq': 5000,
    'save_freq': 10000,
    'gamma': 0.98,
}


env = ManipulatorEnv.ManipulatorEnv(param_dict)
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