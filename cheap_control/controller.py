import time
import datetime
import itertools
from run_learning import *
from params import *

eps = [10, 1, 0.1, 0.01]
gamma = [0.95, 0.98, 0.99]
learning_rate = [0.003, 0.0003, 0.00003]


combos = list(itertools.product(eps, gamma, learning_rate))
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

	params.eps = combo[0]
	params.gamma = combo[1]
	params.learning_rate = combo[2]

	run_learning(params)

	time_left = (time.time() - startime) * (num_combos - curr_combo)
	time_left = str(datetime.timedelta(seconds=time_left))

	curr_combo += 1

