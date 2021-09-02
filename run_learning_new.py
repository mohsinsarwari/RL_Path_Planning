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

from Base_env import Base_env

from stable_baselines3 import SAC
from stable_baselines3.sac import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv

from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import CheckpointCallback

def run_learning():
    policy_kwarg = dict(activation_fn=th.nn.Tanh)
    path = "Base_Logs"
    #set hyperparameters to be used in training
    #documentation of parameter meaning and Stable Baselines 3 SAC implementation
    #https://stable-baselines3.readthedocs.io/en/master/modules/sac.html
    #policy_kwarg_lst = [dict(activation_fn=th.nn.Tanh, net_arch=[64, 64, 64])

    # The algorithms require a vectorized environment to run
    env = Base_env()

    gamma=0.1

    model = SAC(MlpPolicy,
                env,
                gamma = gamma,
                use_sde = True,
                policy_kwargs=policy_kwarg,
                verbose = 1,
                device='cuda',
                )

    #execute learning   
    model.learn(total_timesteps=2000000)
    
    #command to run tensorboard from command prompt
    #tensorboard --logdir=C:/Users/mkest/Dynamics_RL/non_min_PVTOL_exp/ --host localhost
    #model.save(path+"/last_epoc_save_2M")
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)

    print("out")
    
    env.render()


if __name__=="__main__":

    run_learning()

    






