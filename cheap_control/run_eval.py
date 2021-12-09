%matplotlib inline

import os
import gym
import csv
import ast
import numpy as np
import torch as th
from io import StringIO
from datetime import datetime
import json
import pickle

from envs.classic_control import *

from stable_baselines3 import SAC
import matplotlib
import matplotlib.pyplot as plt
from run_learning import run_learning


env = PvtolEnv()

def evaluate(model, env, param_dict):
    done = False
    for i_episode in range(20):
    	observation = env.reset()
	    while not done:
	        action, _states = model.predict(obs)
	        obs, rewards, done, info = test_env.step(action)
	        env.render()

    env.close()
    
best_model = SAC.load(os.path.join(path, "best_model"))

total_timesteps = param_dict['total_timesteps']

latest_model = SAC.load(os.path.join(path, "rl_model_{}_steps".format(total_timesteps)))

param_dict