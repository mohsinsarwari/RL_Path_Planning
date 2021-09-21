from run_learning import run_learning
from datetime import datetime
import itertools
import numpy as np
import torch as th
import os
import matplotlib.pyplot as plt

from Base_env import Base_env
from Reference_env import Reference_env
from RL_env import RL_env

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
cost_weights_sweep = [[20, 10, 0.1], [10, 10, 0.1], [5, 10, 0.1], [1, 10, 0.1], [0.1, 10, 0.1]]
#cost_weights_sweep = [[10, 10, 0.1], [10, 5, 0.1]]

#2nd sweeping parameter: gamma
gamma_sweep = [0.90, 0.92, 0.94, 0.96, 0.98]
#gamma_sweep = [0.1, 0.2]

#Set rest of parameters:

#dynamical system
b=-2

#reference system
internal_matrix = [[0, -1], [1, 0]]
path_matrix = [0, 1]

#mediator
dt = 0.1
total_time = 5
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

num_rows = len(cost_weights_sweep)

num_columns = len(gamma_sweep)

f, ax = plt.subplots(num_rows*2, num_columns*3, sharex=True, sharey=True, figsize=(40, 40))
f.suptitle("Gamma vs Cost_weights")

for i in range(len(cost_weights_sweep)):
	for j in range(len(gamma_sweep)):

		gamma = gamma_sweep[j]
		cost_weights = cost_weights_sweep[i]

		param_dict['gamma'] = gamma
		param_dict['cost_weights'] = cost_weights

		model, env = run_learning(param_dict, rootdir, "{}_{}".format(gamma, cost_weights))

		path = os.path.join(rootdir, "{}_{}".format(gamma, cost_weights))

		# Execute Evaluation
		print("Executing Evaluation...")

		test_sizes = [0.2, 1, 3]

		for k in range(3):

			size = test_sizes[k] 

			test_dynamical_env = Base_env(b=b, test=True, initial_state=np.array([0, 0, 0, 0]))
			test_reference_env = Reference_env(internal_matrix, path_matrix, test=True, initial_state=np.array([size, size, 0, 0]))
			test_env = RL_env(test_dynamical_env, test_reference_env, total_time, dt, cost_weights, path)

			obs = test_env.reset()
			done = False
			while not done:
				action, _states = model.predict(obs)
				obs, rewards, done, info = test_env.step(action)

			times, learned, desired, zero = test_env.render()

			ax[2*i, 3*j + k].plot(times, learned, label='learned')
			ax[2*i, 3*j + k].plot(times, desired, label='desired')
			ax[2*i, 3*j + k].set_title("{}_{}_{}".format(gamma, cost_weights, size))
			ax[2*i, 3*j + k].legend()

			ax[2*i + 1, 3*j + k].plot(times, zero)
			ax[2*i + 1, 3*j + k].set_title("{}_{}_{} zero".format(gamma, cost_weights, size))


f.savefig(os.path.join(rootdir, "sweep.png"))


print("All done!")


