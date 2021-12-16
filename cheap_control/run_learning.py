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
import pickle

from envs.classic_control import *

from stable_baselines3 import SAC
from stable_baselines3.sac import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import CheckpointCallback


def run_learning(param_dict):

    path = os.path.join("./Runs", param_dict["folder"])

    try:
        os.mkdir(path)
    except FileExistsError:
        print("Overriding folder ", param_dict["folder"])

    #save param_dict to a csv for later reference
    with open(os.path.join(path, "param_dict.pkl"), 'wb') as f:
        pickle.dump(param_dict, f, pickle.HIGHEST_PROTOCOL)
            
    #Make Envs
    env = quadrotor.QuadrotorEnv(param_dict)
    eval_env = quadrotor.QuadrotorEnv(param_dict)

    #create callback function to occasionally evaluate the performance
    #of the agent throughout training
    eval_callback = EvalCallback(eval_env,
                             best_model_save_path=path,
                             log_path=path,
                             eval_freq=param_dict["eval_freq"],
                             deterministic=True,
                             render=False)

    save_callback = CheckpointCallback(save_freq=param_dict["save_freq"], 
                                        save_path=path,
                                        name_prefix='rl_model')
    
    #create list of callbacks that will be chain-called by the learning algorithm
    callback = [eval_callback, save_callback]

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
                tensorboard_log=path
                )


    # Execute learning 
    print("Executing Learning...")  
    model.learn(total_timesteps=param_dict["total_timesteps"], callback=callback, tb_log_name="tb_log")

    print("Done running learning")


if __name__=="__main__":

    param_dict = {
        #path info
        'folder': "Quadrotor_Test",
        'description': "Testing out Quadrotor Simulation",
        #shared params
        'dt': 0.05,
        #RL_env parameters
        'total_time': 1,
        'total_timesteps': 50000,
        #model parameters
        'policy_kwarg': dict(activation_fn=th.nn.Tanh),
        'eval_freq': 5000,
        'save_freq': 10000,
        'gamma': 0.98,
    }

    run_learning(param_dict)