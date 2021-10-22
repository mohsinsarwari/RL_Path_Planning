from run_learning import run_learning
from datetime import datetime
import itertools
import numpy as np
import torch as th
import os
import matplotlib
matplotlib.use('TKAgg')

from Base_env import Base_env
from Reference_env import Reference_env
from RL_env import RL_env

# This file is designed to hyperparameter sweep over two parameters in one go (so that the results can be plotted in a grid)


#CL to run in background: nohup python /home/mohsin/research/RL/sweep.py > /home/mohsin/research/RL/running.log &
#Find PID: ps ax | grep sweep.py
#Kill PID: kill PID
#Or just kill directly: pkill -f sweep.py

print("Started Sweep")

param_dict = {
    #path info
    'folder': "Calibration_test_min_path",
    'description': "Testing out new setup running min system only weighting path",
    #shared params
    'dt': 0.1,
    'init_low': -3,
    'init_high': 3,
    'test': False,
    #RL_env parameters
    'total_time': 10,
    'total_timesteps': 100000,
    'cost_weights': [1, 0, 0],
    #base env parameters
    'b' : -2,
    'action_high': 4,
    'action_low': -4,
    'initial_state_dynamic': [1, 1],
    #reference env parameters
    'internal_matrix': [[0, -1], [1, 0]],
    'path_matrix': [0, 1],
    'initial_state_reference': [1, 1],
    #model parameters
    'policy_kwarg': dict(activation_fn=th.nn.Tanh),
    'eval_freq': 50000,
    'save_freq': 10000,
    'gamma': 0.98,
}

run_learning(param_dict)

print("1")

param_dict["folder"] = "Calibration_test_nonmin_path"
param_dict["description"] = "Testing out new setup running nonmin system (b=0.5) only weighting path"
param_dict["cost_weights"] = [1, 0, 0]
param_dict["b"] = 0.5

run_learning(param_dict)

print("2")

param_dict["folder"] = "Calibration_test_min_zero"
param_dict["description"] = "Testing out new setup running min system only weighting zero"
param_dict["cost_weights"] = [0, 1, 0]
param_dict["b"] = -2

run_learning(param_dict)

print("3")

param_dict["folder"] = "Calibration_test_nonmin_zero"
param_dict["description"] = "Testing out new setup running nonmin system (b=0.5) only weighting zero"
param_dict["cost_weights"] = [0, 1, 0]
param_dict["b"] = 0.5

run_learning(param_dict)

print("4")

param_dict["folder"] = "Calibration_test_min_input"
param_dict["description"] = "Testing out new setup running nonmin system (b=0.5) only weighting input"
param_dict["cost_weights"] = [0, 0, 1]
param_dict["b"] = -2

run_learning(param_dict)

print("5")

param_dict["folder"] = "Calibration_test_nonmin_input"
param_dict["description"] = "Testing out new setup running nonmin system (b=0.5) only weighting input"
param_dict["cost_weights"] = [0, 0, 1]
param_dict["b"] = 0.5

run_learning(param_dict)

print("6")

param_dict["folder"] = "Calibration_test_min_blend"
param_dict["description"] = "Testing out new setup running min system blending weight (1, 0.2, 0)"
param_dict["cost_weights"] = [1, 0.2, 0]
param_dict["b"] = -2

run_learning(param_dict)

print("7")

param_dict["folder"] = "Calibration_test_nonmin_blend"
param_dict["description"] = "Testing out new setup running nonmin system (b=0.5) blending weight (1, 0.2, 0)"
param_dict["cost_weights"] = [1, 0.2, 0]
param_dict["b"] = 0.5

run_learning(param_dict)

print("8")