#! /usr/bin/python 
"""
@Mohsin

This is the file that will train the model. 

Configure the model and environment parameters here.

@params param_dict: dictionary of parameters

@params root_path: folder where this training should be saved

@params folder_name: new folder in root_path we want to save to

@return model, env
"""

import os
import gym
import csv
import numpy as np
import torch as th
from io import StringIO
from datetime import datetime
import json

from Base_env import Base_env
from Reference_env import Reference_env
from RL_env import RL_env

from stable_baselines3 import SAC
from stable_baselines3.sac import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure

from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from TrainingRewardCallback import TrainingRewardCallback


def run_learning(param_dict, root_path, folder_name):

    # Make log path

    log_path = os.path.join(root_path, folder_name)

    try:
        os.mkdir(log_path)
    except FileExistsError:
        print("Overriding folder ", folder_name)

    # Unwrap param_dict
    dt = param_dict['dt']

    #Reference env parameters
    internal_matrix = param_dict['internal_matrix']
    path_matrix = param_dict['path_matrix']

    #Dynamic env parameters
    b = param_dict['b']

    #RL env parameters
    total_time = param_dict['total_time']
    cost_weights = param_dict['cost_weights']

    #model parameters
    policy_kwarg = param_dict['policy_kwarg']
    gamma = param_dict['gamma']
    total_timesteps = param_dict['total_timesteps']
    eval_freq = param_dict['eval_freq']
    save_freq = param_dict['save_freq']

    #info needed for storing the dictionary of relevant parameters in .csv file
    fields = param_dict.keys()
    savename = "param_save.csv" 

    #save param_dict to a csv for later reference
    save_path = os.path.join(path, savename)
    with open(save_path, 'w') as csvfile: 
        # creating a csv dict writer object 
        writer = csv.DictWriter(csvfile, fieldnames = fields)    
        # writing headers (field names) 
        writer.writeheader()  
        # writing data rows 
        writer.writerow(param_dict) 
            

    #Make Envs
    dynamical_env = Base_env(b=b, dt=dt)

    reference_env = Reference_env(internal_matrix=internal_matrix, path_matrix=path_matrix, dt=dt)

    env = RL_env(dynamical_env, reference_env, total_time, cost_weights, path)
    eval_env = RL_env(dynamical_env, reference_env, total_time, cost_weights, path)
            

    #create callback function to occasionally evaluate the performance
    #of the agent throughout training
    eval_callback = EvalCallback(eval_env,
                             best_model_save_path=path,
                             log_path=path,
                             eval_freq=eval_freq,
                             deterministic=False,
                             render=False)

    trainingreward_callback = TrainingRewardCallback()
    
    #create list of callbacks that will be chain-called by the learning algorithm
    callback = [eval_callback, trainingreward_callback]

    # Make Model
    model = SAC(MlpPolicy,
                env,
                gamma = gamma,
                use_sde = True,
                policy_kwargs=policy_kwarg,
                verbose = 0,
                #device='cuda',
                )

    # Execute learning 
    print("Executing Learning...")  
    model.learn(total_timesteps=total_timesteps, callback=callback)
    print("Done running learning")
    
    return model, env

if __name__=="__main__":

    param_dict = {
        'b' : -2,
        'internal_matrix': [[0, -1], [1, 0]],
        'path_matrix': [0, 1],
        'total_time': 10,
        'dt': 0.1,
        'total_timesteps': 200,
        'policy_kwarg': dict(activation_fn=th.nn.Tanh),
        'eval_freq': 10,
        'save_freq': 10,
        'gamma': 0.9,
        'cost_weights': [10, 10, 1]
    }

    run_learning(param_dict, ".", "Test")

    






