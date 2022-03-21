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
import datetime
import json
import dill as pickle # so that we can pickle lambda functions

from params import *

from stable_baselines3 import SAC, PPO
from stable_baselines3.sac import MlpPolicy

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import CheckpointCallback

BASE_PATH = "./Runs"

def run_learning(params):

    #------------Setup Folders and log file------------------------------

    current_time = datetime.datetime.now()
    folder_name = current_time.strftime("%m_%d_%Y_%H%M%S") + "_" + params.runner

    path = os.path.join(BASE_PATH, folder_name)
    os.mkdir(path)


    #-----------Train each model for the different environments---------

    for env_name, env_params in zip(params.envs.keys(), params.envs.values()):

        if not env_params.run:
            continue

        env_path = os.path.join(path, env_name)
        os.mkdir(env_path)

        models_path = os.path.join(env_path, "models")
        os.mkdir(models_path)

        tensorboard_log = os.path.join(env_path, "tensorboard_log")
        os.mkdir(tensorboard_log)

        env = env_params.env(params)
        env.reset()

        eval_env = env_params.eval_env(params)
        eval_env.reset()

        #create callback function to occasionally evaluate the performance
        #of the agent throughout training
        eval_callback = EvalCallback(eval_env,
                                 best_model_save_path=models_path,
                                 n_eval_episodes=10,
                                 eval_freq=params.eval_freq,
                                 log_path=env_path,
                                 deterministic=True,
                                 render=False)

        save_callback = CheckpointCallback(save_freq=params.save_freq, 
                                            save_path=models_path,
                                            name_prefix='rl_model')

        #create list of callbacks that will be chain-called by the learning algorithm
        callback = [eval_callback] #, save_callback]

        # Make Model
        #command to run tensorboard from command prompt
        if params.algorithm=="SAC":
            model = SAC(MlpPolicy,
                        env,
                        gamma = params.gamma,
                        learning_rate = params.learning_rate,
                        use_sde = params.use_sde,
                        policy_kwargs=params.policy_kwargs,
                        verbose = 1,
                        device="cuda",
                        tensorboard_log = tensorboard_log
                        )
        elif params.algorithm=="PPO":
            model = PPO("MlpPolicy",
                        env,
                        device="cuda",
                        tensorboard_log = tensorboard_log
                        )                

        # Execute learning   
        model.learn(total_timesteps=params.timesteps, callback=callback)

        model.save(os.path.join(models_path, "last_model"))

        params.finished = True

    with open(os.path.join(path, "params.pkl"), 'wb') as pick:
        pickle.dump(params, pick, pickle.HIGHEST_PROTOCOL)

if __name__=="__main__":
    run_learning(params)

