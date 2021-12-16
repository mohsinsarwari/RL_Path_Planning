import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path


class ManipulatorEnv(gym.Env):

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self, param_dict, k1=1, k2=1, k3=1):
        self.dt = param_dict["dt"]
        self.num_steps = param_dict["total_time"] // self.dt
        self.curr_step = 0
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.max_input = 1
        self.min_input = -1
        self.max_state = 5
        self.min_state = -5
        self.viewer = None
        self.done = False

        self.action_space = spaces.Box(low=self.min_input, high=self.max_input, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=self.min_state, high=self.max_state,shape=(4,), dtype=np.float32)

        self.seed()

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
                                self.k1*np.sin(theta) + self.k2*(phi-theta),
                                self.k3*(theta-phi) + u])

        self.state = self.state + (self.dt * derivatives)

        costs = (self.state[1] - 2)**2 + (self.state[0] - 2)**2

        self.curr_step += 1
        if self.curr_step == self.num_steps:
            self.done = True
 
        return self.state, -costs, self.done, {}

    def reset(self):
        self.state = np.zeros(4)
        self.curr_step = 0
        self.done = False
        return self.state

    def render(self, mode="human"):
        screen_width = 400
        screen_height = 400

        world_width = 2.4
        scale = screen_width / world_width
        carty = 200  # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * 0.5)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = (
                -polewidth / 2,
                polewidth / 2,
                polelen - polewidth / 2,
                -polewidth / 2,
            )
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(0.8, 0.6, 0.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(0.5, 0.5, 0.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

            self._pole_geom = pole

        if self.state is None:
            return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        x = self.state
        self.carttrans.set_rotation(x[1])
        self.carttrans.set_translation(200, 200)
        self.poletrans.set_rotation(x[0])

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None