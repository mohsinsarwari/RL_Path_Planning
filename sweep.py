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
rootdir = "MAIN_Sweep_Run"+"_{}".format(cur_time)

#create new path in log folder for current experiment
os.mkdir(rootdir)

print("Starting Sweep")



#1st sweeping parameter: cost_weights
#cost_weights_sweep = [[100, 10, 0.1], [20, 10, 0.1], [1, 10, 0.1], [5, 10, 0.1], [1, 10, 0.1]]
cost_weights_sweep = [[10, 10, 0.1], [10, 5, 0.1]]

#2nd sweeping parameter: gamma
#gamma_sweep = [0.1, 0.3, 0.5, 0.7, 0.9]
gamma_sweep = [0.1, 0.2]

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
total_timesteps=100
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


f_path_small, ax_path_small = plt.subplots(len(cost_weights_sweep), len(gamma_sweep), sharex=True, sharey=True, figsize=(15, 15))
f_zero_small, ax_zero_small = plt.subplots(len(cost_weights_sweep), len(gamma_sweep), sharex=True, sharey=True, figsize=(15, 15))
f_path_small.suptitle("Gamma vs cost_weights Small Amp")
f_zero_small.suptitle("Gamma vs cost_weights Small Amp")

f_path_medium, ax_path_medium = plt.subplots(len(cost_weights_sweep), len(gamma_sweep), sharex=True, sharey=True, figsize=(15, 15))
f_zero_medium, ax_zero_medium = plt.subplots(len(cost_weights_sweep), len(gamma_sweep), sharex=True, sharey=True, figsize=(15, 15))
f_path_medium.suptitle("Gamma vs cost_weights Medium Amp")
f_zero_medium.suptitle("Gamma vs cost_weights Medium Amp")

f_path_large, ax_path_large = plt.subplots(len(cost_weights_sweep), len(gamma_sweep), sharex=True, sharey=True, figsize=(15, 15))
f_zero_large, ax_zero_large = plt.subplots(len(cost_weights_sweep), len(gamma_sweep), sharex=True, sharey=True, figsize=(15, 15))
f_path_large.suptitle("Gamma vs cost_weights Large Amp")
f_zero_large.suptitle("Gamma vs cost_weights Large Amp")

num_run = len(cost_weights_sweep)*len(gamma_sweep)
curr = 1
for i in range(len(cost_weights_sweep)):
	for j in range(len(gamma_sweep)):

		gamma = gamma_sweep[j]
		cost_weights = cost_weights_sweep[i]

		param_dict['gamma'] = gamma
		param_dict['cost_weights'] = cost_weights

		print("Running {0} out of {1}: ".format(curr, num_run))

		model, env = run_learning(param_dict, rootdir, "{}_{}".format(gamma, cost_weights))

		# Execute Evaluation
		print("Executing Evaluation...")
		small = 0.2
		test_dynamical_env = Base_env(b=b, test=True, initial_state=np.array([0, 0, 0, 0]))
		test_reference_env = Reference_env(internal_matrix, path_matrix, test=True, initial_state=np.array([small, small, 0, 0]))
		test_env = RL_env(test_dynamical_env_small, test_reference_env_small, total_time, dt, cost_weights, path)

		obs = test_env.reset()
		done = False
		while not done:
			action, _states = model.predict(obs)
			obs, rewards, done, info = env.step(action)

		times, learned, desired, zero = env.render()

		ax_path_small[i, j].plot(times, learned, label='learned')
		ax_path_small[i, j].plot(times, desired, label='desired')
		ax_path_small[i, j].set_title("{}_{}".format(gamma, cost_weights))
		ax_path_small[i, j].legend()

		ax_zero_small[i, j].plot(times, zero)
		ax_zero_small[i, j].set_title("{}_{}".format(gamma, cost_weights))

		medium = 1
		test_dynamical_env = Base_env(b=b, test=True, initial_state=np.array([0, 0, 0, 0]))
		test_reference_env = Reference_env(internal_matrix, path_matrix, test=True, initial_state=np.array([small, small, 0, 0]))
		test_env = RL_env(test_dynamical_env_small, test_reference_env_small, total_time, dt, cost_weights, path)

		obs = test_env.reset()
		done = False
		while not done:
			action, _states = model.predict(obs)
			obs, rewards, done, info = env.step(action)

		times, learned, desired, zero = env.render()

		ax_path_medium[i, j].plot(times, learned, label='learned')
		ax_path_medium[i, j].plot(times, desired, label='desired')
		ax_path_medium[i, j].set_title("{}_{}".format(gamma, cost_weights))
		ax_path_medium[i, j].legend()

		ax_zero_medium[i, j].plot(times, zero)
		ax_zero_medium[i, j].set_title("{}_{}".format(gamma, cost_weights))

		large = 3
		test_dynamical_env = Base_env(b=b, test=True, initial_state=np.array([0, 0, 0, 0]))
		test_reference_env = Reference_env(internal_matrix, path_matrix, test=True, initial_state=np.array([small, small, 0, 0]))
		test_env = RL_env(test_dynamical_env_small, test_reference_env_small, total_time, dt, cost_weights, path)

		obs = test_env.reset()
		done = False
		while not done:
			action, _states = model.predict(obs)
			obs, rewards, done, info = env.step(action)

		times, learned, desired, zero = env.render()

		ax_path_large[i, j].plot(times, learned, label='learned')
		ax_path_large[i, j].plot(times, desired, label='desired')
		ax_path_large[i, j].set_title("{}_{}".format(gamma, cost_weights))
		ax_path_large[i, j].legend()

		ax_zero_large[i, j].plot(times, zero)
		ax_zero_large[i, j].set_title("{}_{}".format(gamma, cost_weights))

		curr += 1


f_path_small.savefig(os.path.join(rootdir, "sweep_path_small.png"))
f_zero_small.savefig(os.path.join(rootdir, "sweep_zero_small.png"))

f_path_medium.savefig(os.path.join(rootdir, "sweep_path_medium.png"))
f_zero_medium.savefig(os.path.join(rootdir, "sweep_zero_medium.png"))

f_path_large.savefig(os.path.join(rootdir, "sweep_path_large.png"))
f_zero_large.savefig(os.path.join(rootdir, "sweep_zero_large.png"))

print("All done!")


