from run_learning import run_learning
from datetime import datetime
import itertools
import numpy as np
import torch as th
import os
import matplotlib.pyplot as plt

# This file is designed to hyperparameter sweep over two parameters in one go (so that the results can be plotted in a grid)


#CL to run in background: nohup python /home/mohsin/research/RL/sweep.py > /home/mohsin/research/RL/running.log &
#Find PID: ps ax | grep sweep.py
#Kill PID: kill PID
#Or just kill directly: pkill -f sweep.py


cur_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
rootdir = "MAIN_Sweep_Run"+f"_{cur_time}"

#create new path in log folder for current experiment
os.mkdir(rootdir)

print("Starting Sweep")



#1st sweeping parameter: cost_weights
cost_weights_sweep = [[100, 10, 0.1], [20, 10, 0.1], [1, 10, 0.1], [5, 10, 0.1], [1, 10, 0.1]]
#cost_weights_sweep = [[10, 10, 0.1], [10, 5, 0.1]]

#2nd sweeping parameter: gamma
gamma_sweep = [0.1, 0.3, 0.5, 0.7, 0.9]
#gamma_sweep = [0.1, 0.2]

#Set rest of parameters:

#dynamical system
b=-2

#reference system
internal_matrix = [[0, -1], [1, 0]]
path_matrix = [0, 1]

#mediator
dt = 0.1
total_time = 10
# path, zero, input
#cost_weights = [10, 10, 0.1]

#model
#gamma=0.5
total_timesteps=100000
eval_freq=total_timesteps//3
save_freq=total_timesteps//3
policy_kwarg = dict(activation_fn=th.nn.Tanh)

param_dict = {
	'b' : b,
	'internal_matrix': internal_matrix,
	'path_matrix': path_matrix,
	'total_time': total_time,
	'dt': dt,
	'total_timesteps': total_timesteps,
	'policy_kwarg': policy_kwarg,
	'eval_freq': eval_freq,
	'save_freq': save_freq,
}

plt.figure(1)
f_path, ax_path = plt.subplots(len(cost_weights_sweep), len(gamma_sweep), sharex=True, sharey=True, figsize=(15, 15))
f_zero, ax_zero = plt.subplots(len(cost_weights_sweep), len(gamma_sweep), sharex=True, sharey=True, figsize=(15, 15))
f_path.suptitle("Gamma vs cost_weights")
f_zero.suptitle("Gamma vs cost_weights")

num_run = len(cost_weights_sweep)*len(gamma_sweep)
curr = 1
for i in range(len(cost_weights_sweep)):
	for j in range(len(gamma_sweep)):

		gamma = gamma_sweep[j]
		cost_weights = cost_weights_sweep[i]

		param_dict['gamma'] = gamma
		param_dict['cost_weights'] = cost_weights

		print("Running {0} out of {1}: ".format(curr, num_run))


		times, learned, desired, zero = run_learning(param_dict, rootdir, "{}_{}".format(gamma, cost_weights))


		plt.figure(1)
		ax_path[i, j].plot(times, learned, label='learned')
		ax_path[i, j].plot(times, desired, label='desired')
		ax_path[i, j].set_title("{}_{}".format(gamma, cost_weights))
		ax_path[i, j].legend()

		ax_zero[i, j].plot(times, zero)
		ax_zero[i, j].set_title("{}_{}".format(gamma, cost_weights))

		curr += 1


f_path.savefig(os.path.join(rootdir, "sweep_path.png"))
f_zero.savefig(os.path.join(rootdir, "sweep_zero.png"))

print("All done!")


