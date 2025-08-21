import numpy as np

class NonePolicy:
    """A do-nothing policy that always outputs zero accelerations.

    This represents the "none" strategy. It ignores the observation
    and returns a fixed action of (0, 0).
    """
    def __init__(self):
        pass

    def predict(self, obs):
        return np.array([0.0, 0.0], dtype=np.float32), None


class RandomPolicy:
    """A random policy that outputs actions uniformly in [-1, 1].

    This represents the "random" strategy. It ignores the observation
    and returns random accelerations for both phi and psi.
    """
    def __init__(self):
        pass
    
    def predict(self, obs):
        return np.random.uniform(low=-1.0, high=1.0, size=2).astype(np.float32), None

class FFMPolicy:
    """FFM (Forced Frequency Motion) policy.

    Adapted from:
      "Initial phase and frequency modulations of pumping a playground swing"
      (https://journals.aps.org/pre/pdf/10.1103/PhysRevE.107.044203).
    
    This policy starts moving the swinger's torso and legs after the swing
    passes through the first timing condition. It moves them sinusoidally
    with a fixed frequency `omega` and a phase offset `alpha`.
    
    This implementation replicates the core ideas from the referenced paper, but may
    not fully achieve optimal swing pumping due to the combined effects of numerical
    drift and environmental clipping.
    """
    def __init__(self, env, omega_const=2.4, alpha = np.radians(-20)):
        self.phi0 = env.unwrapped.phi0
        self.psi0 = env.unwrapped.psi0
        self.omega = omega_const
        self.alpha = alpha
        self.t = 0 
        self.dt = env.unwrapped.dt
        self.max_body_accel = env.unwrapped.max_body_accel
    def predict(self, obs):        
        phi_ddot = -self.phi0 * self.omega ** 2 * np.sin(self.omega * self.t - self.alpha) / self.max_body_accel
        psi_ddot = -self.psi0 * self.omega ** 2 * np.sin(self.omega * self.t - self.alpha) / self.max_body_accel
        self.t += self.dt
        return np.array([phi_ddot, psi_ddot]), None