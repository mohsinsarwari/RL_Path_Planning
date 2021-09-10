#! /usr/bin/python 
"""
Created on Mon Sep 21 13:56:31 2020
#TODO
1.get test logs to save to log directory
2.pass spline parameters in outputs
3.
4.adapt bicycle_gym_env to new format
5.integrate option to choose environment
6.jet engine example?
@author: mkest
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

from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import CheckpointCallback

def run_learning(param_dict, root_path, folder_name):

    path = os.path.join(root_path, folder_name)

    try:
        os.mkdir(path)
    except FileExistsError:
        print("Overriding folder ", folder_name)

    # unwrap param_dict

    b = param_dict['b']
    internal_matrix = param_dict['internal_matrix']
    path_matrix = param_dict['path_matrix']
    total_time = param_dict['total_time']
    dt = param_dict['dt']
    cost_weights = param_dict['cost_weights']
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
    dynamical_env = Base_env(b=b)

    reference_env = Reference_env(internal_matrix, path_matrix)

    env = RL_env(dynamical_env, reference_env, total_time, dt, cost_weights, path)
    eval_env = RL_env(dynamical_env, reference_env, total_time, dt, cost_weights, path)
            

    #create callback function to occasionally evaluate the performance
    #of the agent throughout training
    eval_callback = EvalCallback(eval_env,
                             best_model_save_path=path,
                             log_path=path,
                             eval_freq=eval_freq,
                             deterministic=False,
                             render=False)
    
    #saves a copy of the current agent every save_freq times steps
    checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path=path,
                             name_prefix='rl_model')
    
    #create list of callbacks that will be chain-called by the learning algorithm
    callback = [eval_callback, checkpoint_callback]

    # Make Model

    model = SAC(MlpPolicy,
                env,
                gamma = gamma,
                use_sde = True,
                policy_kwargs=policy_kwarg,
                verbose = 1,
                device='cuda',
                )

    # Execute learning 
    print("Executing Learning...")  
    model.learn(total_timesteps=total_timesteps, callback=callback)
    
    # Execute Evaluation
    print("Executing Evaluation...")
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)

    print("Done running learning")
    
    return env.render()


if __name__=="__main__":

    param_dict = {
        'b' : -2,
        'internal_matrix': [[0, -1], [1, 0]],
        'path_matrix': [0, 1],
        'total_time': 10,
        'dt': 0.1,
        'total_timesteps': 2000,
        'policy_kwarg': dict(activation_fn=th.nn.Tanh),
        'eval_freq': 1000,
        'save_freq': 1000,
        'gamma': 0.9,
        'cost_weights': [10, 10, 1]
    }

    run_learning(param_dict, ".", "Test")

    






