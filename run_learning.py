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
from Reference_env import Reference_env
from RL_env import RL_env

from stable_baselines3 import SAC
from stable_baselines3.sac import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv

from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import CheckpointCallback

def run_learning():

    # Set name of Folder to save to
    folder = "Test_WO_Input"

    try:
        os.mkdir(folder)
    except FileExistsError:
        print("Overriding folder ", folder)


    # Setup Dynamical System
    dynamical_env = Base_env(b=-2)


    # Setup Reference System
    internal_matrix = [[0, -1], [1, 0]]
    path_matrix = [0, 1]
    reference_env = Reference_env(internal_matrix, path_matrix)

    # Setup Mediator
    dt = 0.1
    total_time = 10
    # path, input, zero
    cost_weights = [10, 10, 0.1]
    env = RL_env(dynamical_env, reference_env, total_time, dt, cost_weights, folder)

    # Setup Model Parameters

    policy_kwarg = dict(activation_fn=th.nn.Tanh)
    gamma=0.5
    total_timesteps=50000

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
    model.learn(total_timesteps=total_timesteps)
    
    # Execute Evaluation
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
    
    env.render()


if __name__=="__main__":

    run_learning()

    






