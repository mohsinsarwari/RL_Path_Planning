import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path


class Manipulator(gym.Env):
    """
    Description:
        Box with rod in center. Box is connected to rod with springs.

    Source:
        ???

    Observation:
        Type: Box(4)
        Num     Observation               Min                     Max
        0       theta                    -2pi                     2pi
        1       phi                      -2pi                     2pi
        2       theta-dot                ??                       ??
        3       phi-dot                  ??                       ??

    Actions:
        Type: Box(1)
        Num   Action
        0     Torque on Box

    Reward:
        Cost function part of params

    Starting State:
        All observations are assigned a uniform random value in [-1..1]

    Episode Termination:
        ????
        Time based
    """

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self):
        pass

    def set_params(self, params):
        self.params = params
        self.action_space = spaces.Box(low=params.min_input, high=params.max_input, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=params.min_state, high=params.max_state,shape=(4,), dtype=np.float32)
        self.num_steps = self.params.total_time // self.params.dt
        params.seed = self.seed()
        self.viewer = None
        self.done = False

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        u = u[0]
        theta = self.state[0]
        phi = self.state[1]
        theta_dot = self.state[2]
        phi_dot = self.state[3]

        derivatives = np.array([theta_dot, phi_dot, 
                                self.params.k1*np.sin(theta) + self.params.k2*(phi-theta),
                                self.params.k3*(theta-phi) + u])

        self.state = self.state + (self.params.dt * derivatives)

        costs = self.params.cost_fn(self.state_to_dict(self.state, u))

        self.curr_step += 1
        self.done = bool(
            self.curr_step == self.num_steps)
 
        return self.state, -costs, self.done, {}

    def reset(self):
        self.state = np.random.uniform(self.params.init_low, self.params.init_high, (4,))
        self.curr_step = 0
        self.done = False
        return self.state

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

    def state_to_dict(self, state, u):
        vals = dict()
        vals["u"] = u
        vals["theta"] = state[0]
        vals["phi"] = state[1]
        vals["theta_dot"] = state[2]
        vals["phi_dot"] = state[3]
        return vals