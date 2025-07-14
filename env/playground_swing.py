from os import path

import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled

class PlaygroundSwingEnv(gym.Env):

    def __init__(self, render_mode: str | None = None, g=9.8):

        self.dt = 0.05
        self.g = g
        
        # drag coefficient
        self.k = 0.4
        self.k_prime = 7/3

        # length and mass of swing
        self.L = 1.81
        self.M0 = 2

        # mass of swinger
        self.m1 = 31.3
        self.m2 = 12.3
        self.m3 = 6.4
        self.M = self.m1 + self.m2 + self.m3
        # height of swinger
        self.l1 = 0.792
        self.l2 = 0.393
        self.l3 = 0.395
        # position of swinger on sit
        self.a = (self.m2/2 + self.m3)*self.l2/self.M
        self.b = (self.m1 + self.m2/2)*self.l2/self.M

        # range of torso position
        self.phi_mean = np.radians(10)
        self.phi0 = np.radians(30)
        # range of legs position
        self.psi_mean = np.radians(10)
        self.psi0 = np.radians(45)
        # speed of body movement
        self.max_body_speed = 1

        # Moments of inertia
        self.I = (self.M0/3 + self.M) * self.L**2
        self.I_prime = (self.m1*self.l1**2 \
                       +self.m2*self.l2**2 \
                       +self.m3*self.l3**2)/3 \
                       +(self.m1*self.m3 - self.m2**2/4) \
                           *self.l2**2/self.M

        N = (self.M0/2 + self.M)*self.L

        self.render_mode = render_mode

        self.screen_dim = 500
        self.screen = None
        self.clock = None
        self.isopen = True

        # position of swing theta, speed of swing theta_dot (limit full rotation speed of math pendulum), position of torso phi, position of legs psi
        high = np.array([np.pi, np.sqrt(5*self.L*self.g), self.phi0, self.psi0], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        
        self.action_space = spaces.Box(
            low=-self.max_body_speed*self.dt, high=self.max_body_speed*self.dt, shape=(2,), dtype=np.float32
        )

    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering
        costs = angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * (u**2)

        newthdot = thdot + (3 * g / (2 * l) * np.sin(th) + 3.0 / (m * l**2) * u) * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * dt

        self.state = np.array([newth, newthdot])

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return self._get_obs(), -costs, False, False, {}

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if options is None:
            high = np.array([np.radians(-33), 0, 0, 0])
        else:
            # Note that if you use custom reset bounds, it may lead to out-of-bound
            # state/observations.
            
            theta = options.get("theta_init") if "theta_init" in options else np.radians(-33)
            theta_dot = options.get("theta_dot_init") if "theta_dot_init" in options else 0
            phi = options.get("phi_init") if "phi_init" in options else 0
            psi = options.get("psi_init") if "psi_init" in options else 0
            
            theta = utils.verify_number_and_cast(theta)
            theta_dot = utils.verify_number_and_cast(theta_dot)
            phi = utils.verify_number_and_cast(phi)
            psi = utils.verify_number_and_cast(psi)
            
            high = np.array([theta, theta_dot, phi, psi])
        low = -high  # We enforce symmetric limits.
        self.state = self.np_random.uniform(low=low, high=high)
        self.last_u = None

        return np.array([theta, theta_dot, phi, psi], dtype=np.float32), {}

def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi