import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path


class Manipulator(gym.Env):
    """
    Description:
        Box with rod in center. Box is connected to rod with springs.

    State:
        theta, phi, theta_dot, phi_dot

    Observation:
        Type: Box(4)
        Num     Observation               Min                     Max
        0       cos(theta)                -1                       1
        1       sin(theta)                -1                       1
        2       cos(phi)                  -1                       1
        3       sin(phi)                  -1                       1
        4       theta-dot                -100000                100000  
        5       phi-dot                  -100000                100000 

    Actions:
        Type: Box(1)
        Num   Action
        0     Torque on Box

    Reward:

    Starting State:
        All observations are assigned a uniform random value in [-1..1]

    Episode Termination:
        ????
        Time based
    """

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self, params, init=None):
        self.curr_eval = 0
        self.init = init
        self.global_params = params
        self.env_params = params.envs.manipulator
        self.action_space = spaces.Box(low=self.env_params.min_input, high=self.env_params.max_input, shape=(1,), dtype=np.float32)

        high = np.array([1, 1, 1, 1, 100000, 100000])
        self.observation_space = spaces.Box(low=-high, high=high,shape=(6,), dtype=np.float32)

        self.num_steps = self.global_params.total_time // self.global_params.dt
        self.viewer = None
        self.done = False

    def set_init(self, init):
        self.init = init

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        
        theta = self.state[0]
        phi = self.state[1]
        theta_dot = self.state[2]
        phi_dot = self.state[3]
        u = action[0]

        if (self.env_params.integration == "direct"):

            derivatives = np.array([theta_dot, phi_dot, 
                                    self.env_params.k1*np.sin(theta) + self.env_params.k2*(phi-theta), - self.env_params.b1*theta_dot,
                                    self.env_params.k3*(theta-phi) + u - self.env_params.b2*phi_dot
                                    ])

            self.state = self.state + (self.global_params.dt * derivatives)

        elif (self.env_params.integration == "sequential"):

            new_phi_dot = self.env_params.k3*(theta-phi) + u
            new_phi = phi + (self.global_params.dt * new_phi_dot)

            new_theta_dot = self.env_params.k1*np.sin(theta) + self.env_params.k2*(new_phi-theta)
            new_theta = theta + (self.global_params.dt * new_theta_dot)

            self.state = np.array([new_theta, new_phi, new_theta_dot, new_phi_dot])

        if (self.env_params.cost_func == 1):
            costs = self.get_cost1(u)
        elif (self.env_params.cost_func == 2):
            costs = self.get_cost2(u)
        elif (self.env_params.cost_func == 3):
            costs = self.get_cost3(u)
        elif (self.env_params.cost_func == 4):
            costs = self.get_cost4(u)

        self.curr_step += 1

        self.done = bool(
            self.curr_step == self.num_steps)

        return self._get_obs(), -costs, self.done, {}

    def get_cost1(self, u):

        theta = self.angle_normalize(self.state[0])
        theta_dot = self.state[2]

        return (theta**2) + (self.env_params.alpha*(self.global_params.eps**0.5)*(theta_dot**2)) + (self.global_params.eps*(u**2))

    def get_cost2(self, u):

        phi = self.angle_normalize(self.state[1])
        phi_dot = self.state[3]

        return (phi**2) + (self.env_params.alpha*(self.global_params.eps**0.5)*(phi_dot**2)) + (self.global_params.eps*(u**2))

    def get_cost3(self, u):

        theta = self.angle_normalize(self.state[0])
        theta_dot = self.state[2]

        return (theta**2) + (self.global_params.eps*(u**2))

    def get_cost4(self, u):

        phi = self.angle_normalize(self.state[1])
        phi_dot = self.state[3]

        return (phi**2) + (self.global_params.eps*(u**2))

    def reset(self):
        if self.init:
            self.state = self.init
            self.state[0] = self.state[0] + self.state[1]
        else:
            self.state = np.random.uniform(self.env_params.init_low, self.env_params.init_high, (4,))
            self.state[0] = self.state[0] + self.state[1]
        self.curr_step = 0
        self.done = False
        return self._get_obs()

    def _get_obs(self):
        theta, phi, theta_dot, phi_dot = self.state
        return np.array([np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi), theta_dot, phi_dot])

    def render(self, mode="human"):
        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            length = 1
            width = 0.5 / 5

            l, r, t, b = -0.2, 0.2, 0.4, -0.2
            box = rendering.make_polygon([(l, b), (l, t), (r, t), (r, b)])
            box.set_color(0, 0, 0)
            self.box_transform = rendering.Transform()
            box.add_attr(self.box_transform)
            self.viewer.add_geom(box)

            l, r, t, b = -width / 2, width / 2, 0, length
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

        self.box_transform.set_rotation(self.state[1])
        self.pole_transform.set_rotation(self.state[0])


        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def angle_normalize(self, x):
            return abs(((x + np.pi) % (2 * np.pi)) - np.pi)