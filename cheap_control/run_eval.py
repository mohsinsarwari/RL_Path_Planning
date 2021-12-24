import os
import gym
import csv
import ast
import numpy as np
import torch as th
from io import StringIO
from datetime import datetime
import json
import dill as pickle
from params import *

from envs import *

from stable_baselines3 import SAC
import matplotlib
import matplotlib.pyplot as plt

def evaluate(model, env):
	for i_episode in range(20):
		obs = env.reset()
		done = False
		while not done:
			action, _states = model.predict(obs)
			obs, rewards, done, info = env.step(action)
			env.render()

	env.close()

path = "./Runs/EpsilonSweepPendulumTest/pendulum/"
	
best_model = SAC.load(os.path.join(path, "models/eps_0.5/best_model"))

with open(os.path.join(path, "params.pkl"), 'rb') as f:
   params = pickle.load(f)

env = Pendulum.Pendulum()
env.set_params(params.envs.pendulum)

evaluate(best_model, env)