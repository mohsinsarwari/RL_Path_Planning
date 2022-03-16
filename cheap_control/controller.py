import time
import datetime
import itertools
from run_learning import *
from params import *

#TO DO: Setup values to loop over
cost_fn = [3, 4]
eps = [0.1, 0.5, 1, 2, 3, 5]

combos = list(itertools.product(cost_fn, eps))

num_combos = len(combos*params.trials)
curr_combo = 0
time_left = 0
beginning_time = datetime.datetime.now().strftime("%m/%d/%Y_%H:%M:%S")

for combo in combos:

	startime = time.time()

	f = open("./log.txt", "+a")
	f.truncate(0)
	f.write("Started current run at {} \n".format(beginning_time))
	f.write("On Combo: {} out of {} \n".format(curr_combo, num_combos))
	f.write("Estimated Time Left: {} \n".format(time_left))
	f.close()

	#TO DO: Unpack values based on order passed into line 11
	#params.envs.manipulator.integration = combo[0]
	params.envs.manipulator.cost_fn = combo[0]
	params.eps = combo[1]

	for i in np.arange(params.trials):
		run_learning(params)
		time_left = (time.time() - startime) * (num_combos - curr_combo)
		time_left = str(datetime.timedelta(seconds=time_left))
		f = open("./log.txt", "+a")
		f.truncate(0)
		f.write("Started current run at {} \n".format(beginning_time))
		f.write("On Combo: {} out of {} \n".format(curr_combo, num_combos))
		f.write("Estimated Time Left: {} \n".format(time_left))
		f.close()
		curr_combo += 1

	params.trial_id += 1

f = open("./log.txt", "+a")
f.truncate(0)
f.write("Done \n")
f.write("Started: {} \n".format(beginning_time))
f.write("Finished: {} \n".format(datetime.datetime.now().strftime("%m/%d/%Y_%H:%M:%S")))
f.close()
