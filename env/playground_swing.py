from os import path

import numpy as np
from math import sin, cos, sqrt, pi

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled

from utils import RK4_for_2nd_order_ODE

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
        # self.phi_mean = np.radians(10)
        # self.phi0 = np.radians(30)
        self.max_phi = np.radians(40)
        self.min_phi = np.radians(-20)
        # # range of legs position
        # self.psi_mean = np.radians(10)
        # self.psi0 = np.radians(45)
        self.max_psi = np.radians(55)
        self.min_psi = np.radians(-35)
        # body angular acceleration
        self.max_body_accel = 8

        # Moments of inertia
        self.I = (self.M0/3 + self.M) * self.L**2
        self.I_prime = (self.m1*self.l1**2 \
                       +self.m2*self.l2**2 \
                       +self.m3*self.l3**2)/3 \
                       +(self.m1*self.m3 - self.m2**2/4) \
                           *self.l2**2/self.M

        self.N = (self.M0/2 + self.M)*self.L

        # position of swing theta, speed of swing theta_dot (limit full rotation speed of math pendulum), position of torso phi, position of legs psi
        high = np.array([pi, sqrt(5*self.L*self.g), self.max_phi,10, self.max_psi,10], dtype=np.float32)
        low = np.array([-pi, -sqrt(5*self.L*self.g), self.min_phi,-10, self.min_psi,-10], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        self.action_space = spaces.Box(
            low=-self.max_body_accel, high=self.max_body_accel, shape=(2,), dtype=np.float32
        )

    def step(self, u):
        theta, theta_dot, phi, phi_dot, psi, psi_dot = self.state

        g = self.g
        dt = self.dt
        
        k = self.k
        k_prime = self.k_prime
        
        L = self.L
        m1 = self.m1
        m3 = self.m3

        l1 = self.l1
        l3   = self.l3  

        a = self.a
        b = self.b

        I = self.I
        I_prime = self.I_prime

        N = self.N

        # t is "fake" so quantitative method would work
        def eq_theta_ddot(t, theta, theta_dot):
            torque =-N*g*sin(theta) \
                    +m1/2*l1*g*sin(theta+phi) \
                    -m3/2*l3*g*sin(theta+psi) \
                    -m1/2*l1*(2/3*l1 - L*cos(phi) + a*sin(phi))*phi_ddot \
                    -m3/2*l3*(2/3*l3 + L*cos(psi) + b*sin(psi))*psi_ddot \
                    -m1/2*l1*(L*sin(phi) + a*cos(phi))*(2*theta_dot*phi_dot + phi_dot**2) \
                    -m3/2*l3*(-L*sin(psi)+ b*cos(psi))*(2*theta_dot*psi_dot + psi_dot**2) \
                    -k*np.sign(theta_dot)*(theta_dot**2*L**3 + k_prime)
            return torque/((I+I_prime) + m1*l1*(-L*cos(phi) + a*sin(phi)) + m3*l3*(L*cos(psi) + b*sin(psi)))
        
        u = np.clip(u, -self.max_body_accel, self.max_body_accel)
        
        # the higher speed the better 
        reward = theta_dot

        phi_ddot = u[0]
        psi_ddot = u[1]
        
        # # if torque would get body position out of bound then its zero
        if phi_ddot > 0 and phi == self.max_phi:
            phi_ddot=0
        if phi_ddot < 0 and phi == self.min_phi:
            phi_ddot=0
        if psi_ddot > 0 and psi == self.max_psi:
            psi_ddot=0
        if psi_ddot < 0 and psi == self.min_psi:
            psi_ddot=0
                
        phi_dot += phi_ddot*dt
        psi_dot += psi_ddot*dt
               
        if phi_dot > 0 and phi == self.max_phi:
            phi_dot=0
        if phi_dot < 0 and phi == self.min_phi:
            phi_dot=0
        if psi_dot > 0 and psi == self.max_psi:
            psi_dot=0
        if psi_dot < 0 and psi == self.min_psi:
            psi_dot=0
        
        phi = np.clip(phi + phi_dot*dt + (phi_ddot*dt**2)/2, self.min_phi, self.max_phi)
        psi = np.clip(psi + psi_dot*dt + (psi_ddot*dt**2)/2, self.min_psi, self.max_psi)
        
        theta, theta_dot = RK4_for_2nd_order_ODE(eq_theta_ddot, dt, dt, theta, theta_dot)

        self.state = np.array([theta, theta_dot, phi, phi_dot, psi, psi_dot])


        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return self.state, reward, False, False, {}

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if options is None:
            theta = np.radians(33)
            theta_dot = 0
            phi = self.max_phi
            phi_dot = 0
            psi = self.max_psi
            psi_dot = 0
        else:            
            theta = options.get("theta_init") if "theta_init" in options else np.radians(33)
            theta_dot = options.get("theta_dot_init") if "theta_dot_init" in options else 0
            phi = options.get("phi_init") if "phi_init" in options else self.max_phi
            phi_dot = options.get("phi_dot_init") if "phi_dot_init" in options else 0
            psi = options.get("psi_init") if "psi_init" in options else self.max_psi
            psi_dot = options.get("psi_dot_init") if "psi_dot_init" in options else 0
            
            theta = utils.verify_number_and_cast(theta)
            theta_dot = utils.verify_number_and_cast(theta_dot)
            phi = utils.verify_number_and_cast(phi)
            phi_dot = utils.verify_number_and_cast(phi_dot)
            psi = utils.verify_number_and_cast(psi)
            psi_dot = utils.verify_number_and_cast(psi_dot)
            
        high = np.array([theta, theta_dot, phi, phi_dot, psi, psi_dot])
        low = np.array([-theta, -theta_dot, self.min_phi, -phi_dot, self.min_psi, -psi_dot])
        self.state = self.np_random.uniform(low=low, high=high)
        self.last_u = None

        return np.array(self.state, dtype=np.float32), {}