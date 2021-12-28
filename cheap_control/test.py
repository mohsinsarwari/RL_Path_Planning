from envs import *
import os
import gym
import csv
import numpy as np
import torch as th
from io import StringIO
from datetime import datetime
import json
import pickle
from dotmap import DotMap
from params import *

env = Pendulum.Pendulum()
env.set_params(params.envs.pendulum)


for i_episode in range(10):
    obs = env.reset()
    done = False
    i = 0
    while not done:
        i += 1
        action = env.action_space.sample()
        #env.render()
        obs, reward, done, info = env.step(action)
        if done:
            print(i)
env.close()
