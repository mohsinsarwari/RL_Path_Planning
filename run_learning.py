#! /usr/bin/python 
"""
@Mohsin


This is the file that will train the model on the passed in parameters 

Configure the model and environment parameters here.

@params param_dict: dictionary of parameters

@params root_path: folder where this training should be saved

@params folder_name: new folder in root_path we want to save to

@return best_model (as decided by eval callback), env

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


def run_learning(param_dict, root_path, folder_name, tensorboard_log, tb_log_name):

    path = os.path.join(root_path, folder_name)

    try:
        os.mkdir(path)
    except FileExistsError:
        print("Overriding folder ", folder_name)

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
    dynamical_env = Base_env(param_dict)
    reference_env = Reference_env(param_dict)

    env = RL_env(dynamical_env, reference_env, param_dict, path)
    eval_env = RL_env(dynamical_env, reference_env, param_dict, path)
            

    #create callback function to occasionally evaluate the performance
    #of the agent throughout training
    eval_callback = EvalCallback(eval_env,
                             best_model_save_path=path,
                             log_path=path,
                             eval_freq=param_dict["eval_freq"],
                             deterministic=True,
                             render=False)
    
    #create list of callbacks that will be chain-called by the learning algorithm
    callback = [eval_callback]

    # Make Model

    #command to run tensorboard from command prompt
    #tensorboard --logdir=/home/mohsin/research/RL_new/
    model = SAC(MlpPolicy,
                env,
                gamma = param_dict["gamma"],
                use_sde = True,
                policy_kwargs=param_dict["policy_kwarg"],
                verbose = 1,
                device='cuda',
                tensorboard_log=tensorboard_log
                )


    # Execute learning 
    print("Executing Learning...")  
    model.learn(total_timesteps=param_dict["total_timesteps"], callback=callback, tb_log_name=tb_log_name)

    print("Done running learning")

    best_model = SAC.load(os.path.join(path, "best_model"))
    
    return model, env


if __name__=="__main__":

    param_dict = {
        #shared params
        'dt': 0.1,
        'init_low': -5,
        'init_high': 5,
        'test': False,
        #RL_env parameters
        'total_time': 10,
        'total_timesteps': 10000,
        'cost_weights': [10, 10, 1],
        'test_sizes': [0.2, 1, 3],
        #base env parameters
        'b' : -2,
        'action_high': 10,
        'action_low': -10,
        #reference env parameters
        'internal_matrix': [[0, -1], [1, 0]],
        'path_matrix': [0, 1],
        #model parameters
        'policy_kwarg': dict(activation_fn=th.nn.Tanh),
        'eval_freq': 1000,
        'gamma': 0.98,
    }

    run_learning(param_dict, ".", "Run_Test", "Run_Test", "test_run")