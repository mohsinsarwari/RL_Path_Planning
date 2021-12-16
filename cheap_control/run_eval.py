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


def evaluate(model, env):
	for i_episode in range(20):
		obs = env.reset()
		while not env.done:
			action, _states = model.predict(obs)
			obs, rewards, done, info = env.step(action)
			env.render()

	env.close()

path = "./Runs/Quadrotor_Test"
	
best_model = SAC.load(os.path.join(path, "best_model"))

with open(os.path.join(path, "param_dict.pkl"), 'rb') as f:
   param_dict = pickle.load(f)

env = quadrotor.QuadrotorEnv(param_dict)

evaluate(best_model, env)