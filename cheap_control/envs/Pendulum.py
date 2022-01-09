import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path


class Pendulum(gym.Env):
    """
    Description:
        Pendulum model with 1 inputs: torque about its center.

    Source:
        ???

    Observation:
        Type: Box(6)
        Num     Observation               Min                     Max
        2       theta                    -2pi                     2pi
        5       theta-dot                ??                       ??

    Actions:
        Type: Box(2)
        Num   Action
        1     Torque

    Reward:
        Cost function part of params

    Starting State:
        All observations are assigned a uniform random value in [-1..1]

    Episode Termination:
        Time based
    """
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self):
        pass

    def set_params(self, params):
        self.params = params
        self.action_space = spaces.Box(low=params.min_input, high=params.max_input, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=params.min_state, high=params.max_state,shape=(2,), dtype=np.float32)
        self.num_steps = self.params.total_time // self.params.dt
        params.seed = self.seed()
        self.viewer = None
        self.done = False

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):

        theta = self.state[0]
        theta_dot = self.state[1] 
        u = u[0]

        derivatives = np.array([theta_dot, 
                                ((self.params.g / self.params.l) * np.sin(theta)) - (self.params.lam * theta_dot / self.params.m) + (u / (self.params.m * self.params.l ** 2))])


        self.state = [theta, theta_dot] + (self.params.dt * derivatives)

        costs = self.get_cost(u)

        self.curr_step += 1
        self.done = bool(
            self.curr_step == self.num_steps)

        return self.state, -costs, self.done, {}

    def get_cost(self, u):

        theta = self.state[0]
        theta_dot = self.state[1]

        return (theta**2) + (self.params.ep*(u**2))


    def reset(self):
        self.state = np.random.uniform(self.params.init_low, self.params.init_high, (2,))
        self.curr_step = 0
        self.done = False
        return self.state

    def render(self, mode="human"):
        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            length = self.params.l*2
            width = self.params.l / 5

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

        self.pole_transform.set_rotation(self.state[0])

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
