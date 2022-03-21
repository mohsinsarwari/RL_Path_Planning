import time
import datetime
import itertools
from run_learning import *
from params import *

#TO DO: Setup values to loop over
cost_func = [3]
eps = [0.1, 0.5, 1, 2, 3, 5]

combos = list(itertools.product(cost_func, eps))

num_combos = len(combos)

#pick up from where left off
checkpoint = open("./checkpoint.txt", "r")
params.trial_id = int(next(checkpoint))
checkpoint.close()

time_left = 0
beginning_time = datetime.datetime.now().strftime("%m/%d/%Y_%H:%M:%S")

while params.trial_id < num_combos:

	combo = combos[params.trial_id]

	startime = time.time()

	#TO DO: Unpack values based on order passed into line 11
	params.envs.manipulator.cost_func = combo[0]
	params.eps = combo[1]

	log = open("./log.txt", "+a")
	log.truncate(0)
	log.write("Started current run at {} \n".format(beginning_time))
	log.write("On Combo: {} out of {} \n".format(params.trial_id+1, num_combos))
	log.write("Estimated Time Left: {} \n".format(time_left))
	log.write("Current combo started at:  {} \n".format(datetime.datetime.now().strftime("%m/%d/%Y_%H:%M:%S")))
	log.close()

	for i in np.arange(params.trials):
		log = open("./log.txt", "+a")
		log.write("Trial {} of {} \n".format(i+1, params.trials))
		log.close()
		run_learning(params)
		time_left = (time.time() - startime) * (num_combos - params.trial_id)
		time_left = str(datetime.timedelta(seconds=time_left))

	#checkpoint
	params.trial_id += 1
	checkpoint = open("./checkpoint.txt", "w")
	checkpoint.write("{}\n".format(params.trial_id))
	checkpoint.close()

	
checkpoint = open("./checkpoint.txt", "w")
checkpoint.write("{}\n".format(0))
checkpoint.close()

log = open("./log.txt", "+a")
log.truncate(0)
log.write("Done \n")
log.write("Started: {} \n".format(beginning_time))
log.write("Finished: {} \n".format(datetime.datetime.now().strftime("%m/%d/%Y_%H:%M:%S")))
log.close()
