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

	results = dict()

	path = os.path.join(BASE_PATH, folder_name)

	with open(os.path.join(path, "params.pkl"), 'rb') as f:
		params = pickle.load(f)

	for env_name, env_params in zip(params.envs.keys(), params.envs.values()):

		if not (params.envs[env_name].run):
			continue

		models_path = os.path.join(path, env_name + "/models")

		if params.algorithm == "SAC":
			model = SAC.load(os.path.join(models_path, model_name))
		elif params.algorithm == "PPO":
			model = PPO.load(os.path.join(models_path, model_name))

		env_results = dict()
		env_path = os.path.join(path, env_name)

		env = env_params.eval_env(params, init=init)

		evaluations = np.load(os.path.join(env_path, "evaluations.npz"))

		env_results["mean_reward"] = [evaluations["timesteps"], evaluations["results"]]

		actions = []
		states = []

		obs = env.reset()
		done = False
		while not done:
			action, _states = model.predict(obs, deterministic=True)
			actions.append(action[0])
			obs, rewards, done, info = env.step(action)
			states.append(env.state)
			if render:
				env.render()

		env.reset()

		env_results["actions"] = actions
		env_results["states"] = np.array(states)

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

