import time
import datetime
import itertools
from run_learning import *
from params import *

#TO DO: Setup values to loop over
eps = [0.1, 0.5, 1, 2, 3, 5]
cost_func = [1, 2]
use_sde = [True, False]

combos = list(itertools.product(eps, cost_func, use_sde))

num_combos = len(combos)
curr_combo = 1
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
	params.eps = combo[0]
	params.envs.newpendulum.cost_func = combo[1]
	params.use_sde = combo[2]

	run_learning(params)

	time_left = (time.time() - startime) * (num_combos - curr_combo)
	time_left = str(datetime.timedelta(seconds=time_left))

	curr_combo += 1

f = open("./log.txt", "+a")
f.truncate(0)
f.write("Done \n")
f.write("Started: {} \n".format(beginning_time))
f.write("Finished: {} \n".format(datetime.datetime.now().strftime("%m/%d/%Y_%H:%M:%S")))
f.close()