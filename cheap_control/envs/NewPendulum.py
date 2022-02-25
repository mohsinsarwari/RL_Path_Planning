import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path


class Pendulum(gym.Env):
    """
    Description:
        Pendulum model with 1 input: torque about its center.

    Source:
        Gym

    Observation:
        Type: Box(3)
        Num     Observation               Min                     Max
        0       cos(theta)               -1                        1
        1       sin(theta)               -1                        1
        2       theta-dot                -inf                     inf

    State:
        0     theta
        1     theta_dot

    Actions:
        Type: Box(1)
        Num   Action
        0     Torque

    Reward:
        See cost function

    Starting State:
        See init_low/high in params.py

    Episode Termination:
        Time based
    """
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self, params, init=None):
        self.curr_eval = 0
        self.init = init
        high = np.array([1, 1, 100000])
        self.global_params = params
        self.env_params = params.envs.newpendulum
        self.action_space = spaces.Box(low=self.env_params.min_input, high=self.env_params.max_input, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, shape=(3,), dtype=np.float32)

        self.state = np.random.uniform(self.env_params.init_low, self.env_params.init_high, (2,))
        
        self.num_steps = self.global_params.total_time // self.global_params.dt
        self.env_params.seed = self.seed()
        self.viewer = None
        self.done = False

    def set_init(self, init):
        self.init = init

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):

        theta = self.state[0]
        theta_dot = self.state[1] 
        u = action[0]

        # damping term:  - (self.env_params.lam * theta_dot / self.env_params.m) \
        # derivatives = np.array([theta_dot, 
        #                         ((3 * self.env_params.g / (2*self.env_params.l)) * np.sin(theta)) + (3 * u / (self.env_params.m * self.env_params.l**2))])

        # self.state = [theta, theta_dot] + (self.global_params.dt * derivatives)

        newtheta_dot = theta_dot + (((3 * self.env_params.g / (2*self.env_params.l)) * np.sin(theta)) + (3 * u / (self.env_params.m * self.env_params.l**2))) * self.global_params.dt

        newth = theta + newtheta_dot * self.global_params.dt

        self.state = np.array([newth, newtheta_dot])

        if (self.env_params.cost_func == 1):
            costs = self.get_cost1(u)
        elif (self.env_params.cost_func == 2):
            costs = self.get_cost2(u)

        self.curr_step += 1

        self.done = bool(
            self.curr_step == self.num_steps)

        return self._get_obs(), -costs, self.done, {}

    def get_cost1(self, u):

        theta = self.state[0]
        theta_dot = self.state[1]

        return (self.angle_normalize(theta)**2) + (self.global_params.eps*(u**2))

    def get_cost2(self, u):

        theta = self.state[0]
        theta_dot = self.state[1]

        return (self.angle_normalize(theta)**2) + ((self.global_params.eps**0.5)*(theta_dot**2)) + (self.global_params.eps*(u**2))

    def reset(self):
        if self.init:
            self.state = self.init
        else:
            self.state = np.random.uniform(self.env_params.init_low, self.env_params.init_high, (2,))

        self.curr_step = 0
        self.done = False
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

    def render(self, mode="human"):
        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-3, 3, -3, 3)

            length = self.env_params.l * 2
            width = self.env_params.l / 5

            l, r, t, b = -width / 2, width / 2, length, 0
            box = rendering.make_polygon([(l, b), (l, t), (r, t), (r, b)])
            circ0 = rendering.make_circle(width / 2)
            circ1 = rendering.make_circle(width / 2)
            circ1.add_attr(rendering.Transform(translation=(0, length)))
            rod = rendering.Compound([box, circ0, circ1])
            rod.set_color(0.8, 0.3, 0.3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(0.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)

        self.pole_transform.set_rotation(self.state[0])

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def angle_normalize(self, x):
        return abs(((x + np.pi) % (2 * np.pi)) - np.pi)
