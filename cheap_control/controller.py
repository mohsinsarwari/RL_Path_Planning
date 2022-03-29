import time
import datetime
import itertools
import argparse
from run_learning import *
from params import *

def sweep(name):

	checkpoint_name = "./logs/{}_checkpoint.txt".format(name)
	log_name = "./logs/{}_log.txt".format(name)

	#TO DO: Setup values to loop over
	init_max = [1, 2, 3, 4, 5]
	combos = init_max
	num_combos = len(combos)

	#pick up from where left off
	try:
		checkpoint = open(checkpoint_name, "r")
		params.trial_id = int(next(checkpoint))
		checkpoint.close()
	except:
		checkpoint = open(checkpoint_name, "w+")
		checkpoint.write("0\n")
		params.trial_id = 0
		checkpoint.close()

	time_left = 0
	beginning_time = datetime.datetime.now().strftime("%m/%d/%Y_%H:%M:%S")

	while params.trial_id < num_combos:

		startime = time.time()

		#TO DO: Unpack values based on order passed into line 11
		params.envs.pendulum.max_input = combos[params.trial_id]
		params.envs.pendulum.min_input = -combos[params.trial_id]

		log = open(log_name, "w")
		log.write("Started current run at {}\n".format(beginning_time))
		log.write("On Combo: {} out of {}\n".format(params.trial_id+1, num_combos))
		log.write("Estimated Time Left: {}\n".format(time_left))
		log.write("Current combo started at:  {}\n".format(datetime.datetime.now().strftime("%m/%d/%Y_%H:%M:%S")))
		log.close()

		for i in np.arange(params.trials):
			log = open(log_name, "a")
			log.write("Trial {} of {}\n".format(i+1, params.trials))
			log.close()
			run_learning(params)
		
		time_left = (time.time() - startime) * (num_combos - params.trial_id)
		time_left = str(datetime.timedelta(seconds=time_left))

		#checkpoint
		params.trial_id += 1
		checkpoint = open(checkpoint_name, "w")
		checkpoint.write("{}\n".format(params.trial_id))
		checkpoint.close()

	checkpoint = open(checkpoint_name, "w")
	checkpoint.write("0\n")
	checkpoint.close()

	log = open(log_name, "w")
	log.write("Done\n")
	log.write("Started: {}\n".format(beginning_time))
	log.write("Finished: {}\n".format(datetime.datetime.now().strftime("%m/%d/%Y_%H:%M:%S")))
	log.close()

if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-name', '-n', type=str, default=None, help='Name of sweep.  Default: None')
	args = parser.parse_args()
	sweep(args.name)
