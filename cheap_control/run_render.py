import os
import argparse
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

from stable_baselines3 import SAC, PPO
import matplotlib
import matplotlib.pyplot as plt

BASE_PATH = "./Runs"

def render(folder, env_name, model_name):
	with open(os.path.join(os.path.join("./Runs", folder), "params.pkl"), 'rb') as f:
	            params = pickle.load(f)
	env = params.envs[env_name].eval_env(params)
	env.reset()
	models_path = os.path.join(os.path.join("./Runs", folder), env_name + "/models")
	model = SAC.load(os.path.join(models_path, model_name))

	obs = env.reset()
	done = False
	i = 0
	while not done:
	    action, _states = model.predict(obs)
	    obs, rewards, done, info = env.step(action)
	    env.render()
	print("hello")
	env.close()


if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-folder', '-f', type=str, default=None, help='Folder to evaluate.  Default: None')
	parser.add_argument('-env', '-e', type=str, default=None, help='Folder to evaluate.  Default: None')
	parser.add_argument('-model', '-m', type=str, default=None, help='Folder to evaluate.  Default: None')

	args = parser.parse_args()
	render(args.folder, args.env_name, agrs.model_name)
