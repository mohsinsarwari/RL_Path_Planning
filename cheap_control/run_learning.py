#! /usr/bin/python 
"""
@Mohsin

This is the file that will train the model on the passed in parameters 

Configure the model and environment parameters in params.py and then use this to run

"""
import os
import gym
import csv
import numpy as np
import torch as th
from io import StringIO
from datetime import datetime
import json
import dill as pickle # so that we can pickle lambda functions

from params import *

from stable_baselines3 import SAC
from stable_baselines3.sac import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import CheckpointCallback

BASE_PATH = "./Runs"

def run_learning(params, env_params, path):

    models_path = os.path.join(path, "models")
    os.mkdir(models_path)

    tb_log_path = os.path.join(path, "tb_log_path")
    os.mkdir(tb_log_path)

    for eps in params.eps:

        env_params.ep = eps
        subfolder = "eps_{}".format(eps)

        print("-----------------------------------------")
        print("ON " + subfolder)

        model_path = os.path.join(models_path, subfolder)
        os.mkdir(model_path)

        env = env_params.env
        env.set_params(env_params)
        env.reset()

        eval_env = env_params.eval_env
        eval_env.set_params(env_params)
        eval_env.reset()

        #create callback function to occasionally evaluate the performance
        #of the agent throughout training
        eval_callback = EvalCallback(eval_env,
                                 best_model_save_path=model_path,
                                 eval_freq=params.eval_freq,
                                 deterministic=True,
                                 render=False)

        save_callback = CheckpointCallback(save_freq=params.save_freq, 
                                            save_path=model_path,
                                            name_prefix='rl_model')

        #create list of callbacks that will be chain-called by the learning algorithm
        callback = [eval_callback, save_callback]

        # Make Model
        #command to run tensorboard from command prompt
        #tensorboard --logdir=/home/mohsin/research/RL_new/
        model = SAC(MlpPolicy,
                    env,
                    gamma = params.gamma,
                    use_sde = True,
                    policy_kwargs=params.policy_kwargs,
                    verbose = 1,
                    tensorboard_log = tb_log_path
                    )

        # Execute learning 
        print("Executing Learning...")  
        model.learn(total_timesteps=params.timesteps, callback=callback, tb_log_name=subfolder)
        print("Done running learning")

for trial in range(params.num_trials):

    root_path = os.path.join(BASE_PATH, params.run_name + "_Trial" + str(trial))

    try:
        os.mkdir(root_path)
    except FileExistsError:
        print("Overriding folder ", params.run_name)

    for env_name, env_params in zip(params.envs.keys(), params.envs.values()):
        if env_params.run:
            print("Running on ", env_name)
            path = os.path.join(root_path, env_name)
            try:
                os.mkdir(path)
            except FileExistsError:
                print("Overriding folder ", params.run_name)

            with open(os.path.join(path, "params.pkl"), 'wb') as f:
                pickle.dump(params, f, pickle.HIGHEST_PROTOCOL)

            run_learning(params, env_params, path)
