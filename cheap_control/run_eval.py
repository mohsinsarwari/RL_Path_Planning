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

BASE_PATH = "./Runs"

def evaluate(path, model="best_model"):

	results = dict()

    path = os.mkdir(os.path.join(BASE_PATH, folder_name))

    models_path = os.path.join(path, "models")

	model = SAC.load(os.path.join(models_path, model))

	with open(os.path.join(path, "params.pkl"), 'rb') as f:
   		params = pickle.load(f)

    for env_name, env_params in zip(params.envs.keys(), params.envs.values()):

    	env_results = dict()

    	env_path = os.path.join(path, env_name)

        if not env_params.run:
            continue

        env = env_params.eval_env
        env.set_params(env_params)
        env.reset()

        evaluations = np.load(os.path.join(env_path, "evaluations.npz"))

        env_results["mean_reward"] = [evaluations["timesteps"], evaluations["results"]]

        actions = []
        states = []

		obs = env.reset()
		done = False
		while not done:
			action, _states = model.predict(obs)
			actions.append(action[0])
			obs, rewards, done, info = env.step(action)
			states.append(obs[0])

		env_results["actions"] = actions
		env_results["states"] = states

		results[env_name] = env_results

		env.close()

	return results

