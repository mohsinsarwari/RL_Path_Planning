from dotmap import DotMap
from envs import *
import torch as th
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.sac import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import CheckpointCallback

params = DotMap()

dt = 0.01
total_time = 10

#General Params
params.eval_freq = 2500
params.save_freq = 5000
params.total_timesteps = 50000
params.gamma = 0.98
params.policy_kwargs = dict(activation_fn=th.nn.Tanh)
params.eps = [0, 0.5, 1, 5]
params.ep = 0
params.run_name = "EpsilonSweepPendulumTest"

#Env Specific Params

params.envs.pendulum.env = Pendulum.Pendulum() #base env for simulation
params.envs.pendulum.eval_env = Pendulum.Pendulum() #extra env for eval callback
params.envs.pendulum.run = True #if you want run_learning to train on this env
params.envs.pendulum.m = 1 #mass of pendulum
params.envs.pendulum.l = 0.5 #half length of pendulum (to center of mass)
params.envs.pendulum.g = 1 #gravity
params.envs.pendulum.lam = 0.03 #damping coefficient
params.envs.pendulum.max_input = 4
params.envs.pendulum.min_input = -4
params.envs.pendulum.max_state = 10
params.envs.pendulum.min_state = -10
params.envs.pendulum.init_low = [-1, -0.5]
params.envs.pendulum.init_high = [1, 0.5]
params.envs.pendulum.cost_fn = lambda vals:  vals["theta"]**2 + (params.ep * (vals["u"]**2))
params.envs.pendulum.dt = dt
params.envs.pendulum.total_time = total_time

params.envs.quadrotor.env = Quadrotor.Quadrotor()
params.envs.quadrotor.eval_env = Quadrotor.Quadrotor()
params.envs.quadrotor.run = False
params.envs.quadrotor.dt = dt
params.envs.quadrotor.total_time = total_time
params.envs.quadrotor.i_xx = 1
params.envs.quadrotor.m = 1
params.envs.quadrotor.g = 1
params.envs.quadrotor.max_input = np.array([10, 3])
params.envs.quadrotor.min_input = np.array([5, -3])
params.envs.quadrotor.max_state = 30
params.envs.quadrotor.min_state = -30
params.envs.quadrotor.init_low = -1
params.envs.quadrotor.init_high = 1
params.envs.quadrotor.cost_fn = lambda vals:  vals["theta"]**2 + (params.eps * (vals["u1"]**2 + vals["u2"]**2))

params.envs.pvtol.env = Pvtol.Pvtol()
params.envs.pvtol.eval_env = Pvtol.Pvtol()
params.envs.pvtol.run = False
params.envs.pvtol.dt = dt
params.envs.pvtol.total_time = total_time
params.envs.pvtol.eps = 0.01
params.envs.pvtol.m = 1
params.envs.pvtol.g = 2
params.envs.pvtol.max_input = np.array([4, 1])
params.envs.pvtol.min_input = np.array([1, -1])
params.envs.pvtol.max_state = 30
params.envs.pvtol.min_state = -30
params.envs.pvtol.init_low = -1
params.envs.pvtol.init_high = 1
params.envs.pvtol.cost_fn = lambda vals:  vals["theta"]**2 + (params.eps * (vals["u1"]**2 + vals["u2"]**2))

params.envs.manipulator.env = Manipulator.Manipulator()
params.envs.manipulator.eval_env = Manipulator.Manipulator()
params.envs.manipulator.run = False
params.envs.manipulator.dt = dt
params.envs.manipulator.total_time = total_time
params.envs.manipulator.eps = 0.01
params.envs.manipulator.k1 = 1
params.envs.manipulator.k2 = 1
params.envs.manipulator.k3 = 1
params.envs.manipulator.max_input = 4
params.envs.manipulator.min_input = -4
params.envs.manipulator.max_state = 5
params.envs.manipulator.min_state = -5
params.envs.manipulator.init_low = -1
params.envs.manipulator.init_high = 1
params.envs.manipulator.cost_fn = lambda vals:  vals["theta"]**2 + (params.eps * (vals["u"]**2))

params.envs.cartpole.env = Cartpole.Cartpole()
params.envs.cartpole.eval_env = Cartpole.Cartpole()
params.envs.cartpole.run = False
params.envs.cartpole.dt = dt
params.envs.cartpole.total_time = total_time
params.envs.cartpole.eps = 0.01
params.envs.cartpole.g = 1
params.envs.cartpole.mp = 0.5
params.envs.cartpole.mc = 1
params.envs.cartpole.l = 0.5  # actually half the pole's length
params.envs.cartpole.f = 5 # force
params.envs.cartpole.lam = 0.01
params.envs.cartpole.max_input = 2
params.envs.cartpole.min_input = -2
params.envs.cartpole.max_state = 5
params.envs.cartpole.min_state = -5
params.envs.cartpole.init_low = -1
params.envs.cartpole.init_high = 1
params.envs.cartpole.cost_fn = lambda vals:  vals["theta"]**2 + (params.eps * (vals["u"]**2))
