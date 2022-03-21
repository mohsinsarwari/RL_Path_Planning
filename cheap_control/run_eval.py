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

def evaluate(folder_name, model_name="best_model", env=None, init=None, render=False, iterations=10):
	#init = [np.pi, 0]

	results = dict()

	path = os.path.join(BASE_PATH, folder_name)

	with open(os.path.join(path, "params.pkl"), 'rb') as f:
		params = pickle.load(f)

	for env_name, env_params in zip(params.envs.keys(), params.envs.values()):

		if not env_params.run or not env_name==env:
			continue

		models_path = os.path.join(path, env_name + "/models")

		if params.algorithm == "SAC":
			model = SAC.load(os.path.join(models_path, model_name))
		elif params.algorithm == "PPO":
			model = PPO.load(os.path.join(models_path, model_name))

		env_results = dict()
		env_path = os.path.join(path, env_name)

		env = env_params.eval_env(params, init=init)
		#env.set_init(init)

		evaluations = np.load(os.path.join(env_path, "evaluations.npz"))

		env_results["mean_reward"] = [evaluations["timesteps"], evaluations["results"]]

		actions = []
		thetas = []
		phis = []

		obs = env.reset()
		done = False
		while not done:
			action, _states = model.predict(obs, deterministic=True)
			actions.append(action[0])
			obs, rewards, done, info = env.step(action)
			thetas.append(angle_normalize(env.state[0]))
			phis.append(angle_normalize(env.state[1]))
			#thetas.append(env.state[0] % (2*np.pi))
			if render:
				env.render()

		env.reset()

		env_results["actions"] = actions
		env_results["thetas"] = thetas
		env_results["phis"] = phis

		results[env_name] = env_results

		env.close()

	return results

def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi

if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-folder', '-f', type=str, default=None, help='Folder to evaluate.  Default: None')
	#parser.add_argument('-init', '-i', type=str, default=None, help='init')
	args = parser.parse_args()
	if args.folder:
		evaluate(args.folder, render=True)
	else:
		evaluate("02_15_2022_123458_Mohsin", render=True)

