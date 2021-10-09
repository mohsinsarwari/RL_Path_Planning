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


cur_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


# Give Folder Meaningful Name
rootdir = "Testing_nonmin_long_run"

#create new path in log folder for current experiment
try:
	os.mkdir(rootdir)
except FileExistsError:
	pass

print("Starting Sweep")

param_dict = {
    #shared params
    'dt': 0.1,
    'init_low': -5,
    'init_high': 5,
    'test': False,
    #RL_env parameters
    'total_time': 10,
    'total_timesteps': 1500000,
    'cost_weights': [10, 50, 1],
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

sweep_param_1 = [0.98]
sweep_param_1_name = "gamma"
num_rows = len(sweep_param_1)

sweep_param_2 = [0.5, 1]
sweep_param_2_name = "b"
num_columns = len(sweep_param_2)

for i in range(len(sweep_param_1)):
	for j in range(len(sweep_param_2)):

		param_1 = sweep_param_1[i]
		param_2 = sweep_param_2[j]

		param_dict[sweep_param_1_name] = param_1
		param_dict[sweep_param_2_name] = param_2

		print("At learning step")
		best_model, env = run_learning(param_dict, rootdir, "{}".format(param_2), rootdir, "{}".format(param_2))

		path = os.path.join(rootdir, "{}".format(param_2))


print("All done!")


