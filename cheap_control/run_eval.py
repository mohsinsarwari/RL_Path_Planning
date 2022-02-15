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

from stable_baselines3 import SAC
import matplotlib
import matplotlib.pyplot as plt

BASE_PATH = "./Runs"

def evaluate(folder_name, model="best_model", init=None, render=False, iterations=10):

	results = dict()

	path = os.path.join(BASE_PATH, folder_name)

	with open(os.path.join(path, "params.pkl"), 'rb') as f:
		params = pickle.load(f)

	for env_name, env_params in zip(params.envs.keys(), params.envs.values()):

		if not env_params.run:
			continue

		models_path = os.path.join(path, env_name + "/models")

		model = SAC.load(os.path.join(models_path, model))

		env_results = dict()
		env_path = os.path.join(path, env_name)

		env = env_params.eval_env(params)
		env.set_init(init)
		env.reset()

		evaluations = np.load(os.path.join(env_path, "evaluations.npz"))

		env_results["mean_reward"] = [evaluations["timesteps"], evaluations["results"]]

		actions = []
		thetas = []

		obs = env.reset()
		done = False
		
		while not done:
			action, _states = model.predict(obs)
			actions.append(action[0])
			obs, rewards, done, info = env.step(action)
			thetas.append(env.angle_normalize(env.state[0]))
			if render:
				env.render()

		env.reset()

		env_results["actions"] = actions
		env_results["thetas"] = thetas

		results[env_name] = env_results

		env.close()

	return results

if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-folder', '-f', type=str, default=None, help='Folder to evaluate.  Default: None')
	parser.add_argument('-iterations', '-i', type=str, default=10, help='Number of iterations')
	args = parser.parse_args()
	if args.folder:
		evaluate(args.folder, render=True)
	else:
		evaluate("02_14_2022_135802_Mohsin", render=True, iterations=args.iterations)

