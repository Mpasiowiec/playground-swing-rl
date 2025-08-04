import numpy as np
import scipy as sp

class FFMPolicy:
    def __init__(self, env, omega_const=2.4, alpha = np.radians(-20)):
        self.phi0 = env.unwrapped.phi0
        self.psi0 = env.unwrapped.psi0
        self.omega = omega_const
        self.alpha = alpha
        self.t = 0 
        self.dt = env.unwrapped.dt
        self.max_body_accel = env.unwrapped.max_body_accel
        self.wait = True
    def select_action(self, obs):
        theta, theta_dot, _, _, _, _ = obs
        if (theta < 0 and self.t == 0) or (theta < 0 and abs(theta_dot)<1e-6):
            self.wait = False
        if self.wait:
            phi_ddot = 0
            psi_ddot = 0    
        else:
            phi_ddot = -self.phi0 * self.omega ** 2 * np.sin(self.omega * self.t - self.alpha) / self.max_body_accel
            psi_ddot = -self.psi0 * self.omega ** 2 * np.sin(self.omega * self.t - self.alpha) / self.max_body_accel
            self.t += self.dt
        return np.array([phi_ddot, psi_ddot])

class PaperModelPolicy:
    def __init__(self, env, alpha = np.radians(65)):
        self.max_body_accel = env.unwrapped.max_body_accel
        self.alpha = alpha
        self.t = 0
        self.dt = env.unwrapped.dt
        self.phi0 = env.unwrapped.phi0
        self.psi0 = env.unwrapped.psi0
        self.m1 = env.unwrapped.m1
        self.m3 = env.unwrapped.m3
        self.l1 = env.unwrapped.l1
        self.l3 = env.unwrapped.l3
        self.L = env.unwrapped.L
        self.N = env.unwrapped.N
        self.a = env.unwrapped.a
        self.b = env.unwrapped.b
        self.I = env.unwrapped.I
        self.I_prime = env.unwrapped.I_prime
        self.g = env.unwrapped.g
        self.swing_done = False
        self.omega = None
    def select_action(self, obs):
        theta, theta_dot, phi, _, psi, _ = obs
        if theta < 0 and self.t == 0: self.swing_done = True
        if not self.swing_done and theta_dot < 0:
          self.swing_done = True
        if self.swing_done and theta < 0 and theta_dot > 0:
            A = abs(theta)
            I_p = self.m1*self.l1*(-self.L*np.cos(phi) + self.a*np.sin(phi)) \
                    +self.m3*self.l3*(self.L*np.cos(psi) + self.b*np.sin(psi))
            C_s = self.N - self.m1/2*self.l1*np.cos(phi) + self.m3/2*self.l3*np.cos(psi)
            C_c = -self.m1/2*self.l1*np.sin(phi) + self.m3/2*self.l3*np.sin(psi)
            T_n = 4*sp.special.ellipk(np.sin(A/2))*np.sqrt((self.I+self.I_prime+I_p)/(self.g*np.sqrt(C_s**2+C_c**2)))
            self.omega = 2*np.pi/T_n
            self.swing_done = False
        if self.omega is None:
            phi_ddot = 0
            psi_ddot = 0
        else:
            phi_ddot = -self.phi0 * self.omega ** 2 * np.cos(self.omega * self.t - self.alpha) / self.max_body_accel
            psi_ddot = -self.psi0 * self.omega ** 2 * np.cos(self.omega * self.t - self.alpha) / self.max_body_accel
            self.t += self.dt
            print(self.phi0 * self.omega ** 2)
        return np.array([phi_ddot, psi_ddot])
