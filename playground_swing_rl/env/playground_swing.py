import os
import shutil
import numpy as np
from math import sin, cos, pi

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.classic_control import utils

from .utils import RK4_for_2nd_order_ODE

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import imageio

class PlaygroundSwingEnv(gym.Env):

    def __init__(self, render_mode: str | None = None, g=9.8, goal='speed', target_angle=np.radians(45)):

        self.render_mode = render_mode
        self.data = {"theta": [], "theta_dot": [], "phi": [], "psi": [], "t": [], "phi_dot": [], "psi_dot": []}
        
        self.goal = goal
        self.target_angle = target_angle
        self.swing_count = 0 # for goal rotation
        self.last_theta_sign = None
        self.full_rotation_count = 0  # for goal rotation
        
        self.t = 0.0
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
        self.phi_max = self.phi_mean + self.phi0
        self.phi_min = self.phi_mean - self.phi0
        # # range of legs position
        self.psi_mean = np.radians(10)
        self.psi0 = np.radians(45)
        self.psi_max = self.psi_mean + self.psi0
        self.psi_min = self.psi_mean - self.psi0
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
        high = np.array([pi, 100, self.phi_max,15, self.psi_max,15], dtype=np.float32)
        low = np.array([-pi, -100, self.phi_min,-15, self.psi_min,-15], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

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

        u = np.clip(u, -1, 1)
        
        terminated = False
        if self.goal == 'speed':
            reward = theta_dot**2

        elif self.goal == 'angle':
            reward = -abs(abs(theta) - self.target_angle)
            current_sign = np.sign(theta - self.target_angle)
            if self.last_theta_sign is not None and current_sign != 0:
                if current_sign != self.last_theta_sign:
                    self.swing_count += 1
            self.last_theta_sign = current_sign
            if self.swing_count >= 4:
                terminated = True

        elif self.goal == 'rotation':
            reward = theta**2
            current_sign = np.sign(theta)
            if self.last_theta_sign is not None:
                # rotation - thetat going from pi to -pi
                if (self.last_theta_sign > 0 and current_sign < 0) or (self.last_theta_sign < 0 and current_sign > 0):
                    self.full_rotation_count += 1
                    # two rotations with going back to the bottom
                    if self.full_rotation_count == 2 and theta<1e-1:
                        terminated = True
            self.last_theta_sign = current_sign

        else:
            reward = theta_dot**2  # domyślnie, jeśli złe wywołanie  


        phi_ddot = u[0] * self.max_body_accel
        psi_ddot = u[1] * self.max_body_accel
        
        # # if accel or speed would get body position out of bound then its zero
        phi_uper_border = abs(phi - (self.phi_max)) < 1e-6 
        phi_lower_border = abs(phi - (self.phi_min)) < 1e-6
        psi_uper_border = abs(psi - (self.psi_max)) < 1e-6
        psi_lower_border = abs(psi - (self.psi_min)) < 1e-6
        
        if ((phi_ddot > 0 and phi_uper_border) or (phi_ddot < 0 and phi_lower_border)):
            phi_ddot=0
        if ((psi_ddot > 0 and psi_uper_border) or  (psi_ddot < 0 and psi_lower_border)):
            psi_ddot=0
                
        phi_dot += phi_ddot*dt
        psi_dot += psi_ddot*dt
               
        if ((phi_dot > 0 and phi_uper_border) or (phi_dot < 0 and phi_lower_border)):
            phi_dot=0
        if ((psi_dot > 0 and psi_uper_border) or (psi_dot < 0 and psi_lower_border)):
            psi_dot=0
        
        # cliping to observation space
        phi = np.clip(phi + phi_dot*dt + (phi_ddot*dt**2)/2, self.phi_min, self.phi_max)
        psi = np.clip(psi + psi_dot*dt + (psi_ddot*dt**2)/2, self.psi_min, self.psi_max)
        
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
        
        theta, theta_dot = RK4_for_2nd_order_ODE(eq_theta_ddot, dt, dt, theta, theta_dot)

        # adding np.nan observation for non continous plots 
        if abs(theta) > pi:
            self.data['theta'].append(np.nan)
            self.data["theta_dot"].append(self.state[1])
            self.data["phi"].append(self.state[2])
            self.data["psi"].append(self.state[4])
            self.data["phi_dot"].append(self.state[3])
            self.data["psi_dot"].append(self.state[5])
            self.data['t'].append(self.t+self.dt/2)
            
            
            # for keeping thteta in opservation sapce in case of full rotation
            theta = ((theta + pi) % (2 * pi)) - pi
        
        self.state = np.array([theta, theta_dot, phi, phi_dot, psi, psi_dot], dtype=np.float32)
        self.t += self.dt

        if self.render_mode:
            self.render()
            
        self.data["theta"].append(self.state[0])
        self.data["theta_dot"].append(self.state[1])
        self.data["phi"].append(self.state[2])
        self.data["psi"].append(self.state[4])
        self.data["phi_dot"].append(self.state[3])
        self.data["psi_dot"].append(self.state[5])
        self.data["t"].append(self.t)

        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return self.state, reward, terminated, False, {}

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if options is None:
            theta = np.radians(33)
            theta_dot = 0
            phi_dot = 0
            psi_dot = 0
            high = np.array([theta, theta_dot, self.phi_max, phi_dot, self.psi_max, psi_dot])
            low = np.array([-theta, -theta_dot, self.phi_min, -phi_dot, self.psi_min, -psi_dot])
            self.state = self.np_random.uniform(low=low, high=high)
        else:            
            theta = np.radians(options.get("theta")) if "theta" in options else np.radians(-33)
            theta_dot = options.get("theta_dot") if "theta_dot" in options else 0
            phi = options.get("phi") if "phi" in options else self.phi_min
            phi_dot = options.get("phi_dot") if "phi_dot" in options else 0
            psi = options.get("psi") if "psi" in options else self.psi_max
            psi_dot = options.get("psi_dot") if "psi_dot" in options else 0
            
            theta = utils.verify_number_and_cast(theta)
            theta_dot = utils.verify_number_and_cast(theta_dot)
            phi = utils.verify_number_and_cast(phi)
            phi_dot = utils.verify_number_and_cast(phi_dot)
            psi = utils.verify_number_and_cast(psi)
            psi_dot = utils.verify_number_and_cast(psi_dot)
            
            self.state = np.array([theta, theta_dot, phi, phi_dot, psi, psi_dot])
        
        self.swing_count = 0
        self.last_theta_sign = None
        self.full_rotation_count = 0
        self.t = 0.0
        self.data = {"theta": [], "theta_dot": [], "phi": [], "psi": [], "t": [], "phi_dot": [], "psi_dot": []}
        if self.render_mode in ["human", "human-plots"]:
            self.render()

        return np.array(self.state, dtype=np.float32), {}

    def _init_figure(self):
        if self.render_mode in ['human-plots', 'gif-plots']:
            self.fig = plt.figure(figsize=(12, 8))
            gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])
            self.ax_phase = self.fig.add_subplot(gs[0, 0])
            self.ax_theta_phi = self.fig.add_subplot(gs[0, 1])
            self.ax_swing = self.fig.add_subplot(gs[1, :])
            # Set fixed axis limits for phase plot and theta/phi vs time
            # self.ax_phase.set_xlim([-1.4, 1.4])  # theta
            # self.ax_phase.set_ylim([-15, 15])  # theta_dot
            # self.ax_theta_phi.set_xlim([0, 5])   # time (adjust as needed)
            # self.ax_theta_phi.set_ylim([-1.4, 1.4])  # theta/phi
            # Initialize lines for plots
            self.phase_line, = self.ax_phase.plot([], [], 'b-')
            self.theta_line, = self.ax_theta_phi.plot([], [], 'b-', label='theta')
            self.phi_line, = self.ax_theta_phi.plot([], [], 'y-', label='phi')
            self.psi_line, = self.ax_theta_phi.plot([], [], 'm-', label='psi')
            self.ax_theta_phi.legend()
        else:
            self.fig, self.ax_swing = plt.subplots(figsize=(12, 8))
            
        self.ax_swing.set_aspect('equal')
        self.ax_swing.set_xlim([-2.5, 2.5])
        self.ax_swing.set_ylim([-2.5, 0]) # for full rotation need change
        self.ax_swing.set_xlabel('x (m)')
        self.ax_swing.set_ylabel('y (m)')
        self.ax_swing.set_title('Swing', fontsize='medium')
        self.fig.tight_layout()
        # Initialize lines
        self.plot0, = self.ax_swing.plot([], [], color='gray', linewidth=0.5)
        self.plot1, = self.ax_swing.plot([], [], color='black', linewidth=2)
        self.plot2, = self.ax_swing.plot([], [], color='black', linewidth=2)
        self.plot3, = self.ax_swing.plot([], [], color='black', linewidth=2)
        self.cm_history_x = []
        self.cm_history_y = []
        self.alphas = np.linspace(0.1, 1.0, self.cm_dots_trail)
        self.cm_scatter = self.ax_swing.scatter([], [], c='r', s=20)

    def render(self):
        if not hasattr(self, 'fig'):
            self.cm_dots_trail = 50
            self._init_figure()
        if self.render_mode in ["human-plots", "gif-plots"]:
            # Update phase plot
            self.phase_line.set_data(self.data["theta"], self.data["theta_dot"])
            self.ax_phase.relim() # Recompute the data limits based on current artists.
            self.ax_phase.autoscale_view() # Autoscale base on limits
            # Update theta/phi vs time
            self.theta_line.set_data(self.data["t"], self.data["theta"])
            self.phi_line.set_data(self.data["t"], self.data["phi"])
            self.psi_line.set_data(self.data["t"], self.data["psi"])
            self.ax_theta_phi.relim()
            self.ax_theta_phi.autoscale_view()

        s1, s2, p1, p2, p3, p4, CM = self._get_positions(self.state)
        
        # Update line data
        self.plot0.set_data([s1[0], s2[0]], [s1[1], s2[1]])
        self.plot1.set_data([p1[0], p2[0]], [p1[1], p2[1]])
        self.plot2.set_data([p1[0], p3[0]], [p1[1], p3[1]])
        self.plot3.set_data([p2[0], p4[0]], [p2[1], p4[1]])

        # Plot the center of mass as a red dot
        self.cm_history_x.append(CM[0])
        self.cm_history_y.append(CM[1])
        if len(self.cm_history_x) > self.cm_dots_trail:
            self.cm_history_x.pop(0)
            self.cm_history_y.pop(0)
        self.cm_scatter.set_offsets(np.c_[self.cm_history_x, self.cm_history_y])
        curr_len = len(self.cm_history_x)
        alphas = self.alphas[-curr_len:]
        colors = [(1, 0, 0, a) for a in alphas]
        self.cm_scatter.set_facecolor(colors)

        if self.render_mode in ["gif", "gif-plots"]:
                if not hasattr(self, 'frames_dir'):
                    self.frames_dir = "frames_tmp"
                    os.makedirs(self.frames_dir, exist_ok=True)
                    self.frame_counter = 0
                frame_path = os.path.join(self.frames_dir, f"frame_{self.frame_counter:05d}.png")
                self.fig.savefig(frame_path, dpi=72)
                self.frame_counter += 1
        else:
            plt.pause(0.001)

    def save_gif_from_frames(self, filename="swing.gif", duration=0.04):
        if not hasattr(self, 'frames_dir'):
            print("No frames to convert to gif.")
            return
        frames_dir = getattr(self, 'frames_dir', "frames_tmp")
        # Zbierz i posortuj ścieżki do plików klatek
        frame_files = [os.path.join(frames_dir, f) for f in sorted(os.listdir(frames_dir)) if f.endswith('.png')]
        with imageio.get_writer(filename, mode='I', duration=duration) as writer:
            for frame_file in frame_files:
                image = imageio.imread(frame_file)
                writer.append_data(image)
        print(f"GIF saved as {filename} from {len(frame_files)} frames.")
        shutil.rmtree(self.frames_dir)

    def _get_positions(self, state):
        # position of swing's chain's ends
        s1 = np.array((0,0))
        s2 = np.array((self.L * sin(state[0]), -self.L * cos(state[0])))
        # position of the butt
        p1 = np.array((self.L*sin(state[0]) - self.a*cos(state[0]), -self.L*cos(state[0]) - self.a*sin(state[0])))
        # position of knees
        p2 = np.array((self.L*sin(state[0]) + self.b*cos(state[0]), -self.L*cos(state[0]) + self.b*sin(state[0])))
        # position of head
        p3 = p1 + self.l1*np.array((-sin(state[0]+state[2]), cos(state[0]+state[2])))
        # position of feet
        p4 = p2 + self.l3*np.array((sin(state[0]+state[4]), -cos(state[0]+state[4])))
        # center of the mass
        mc1 = (p1+p3)/2
        mc2 = (p1+p2)/2
        mc3 = (p2+p4)/2
        CM = (self.m1*mc1 + self.m2*mc2 + self.m3*mc3 + self.M0*np.array((self.L/2*sin(state[0]), -self.L/2*cos(state[0]))))/(self.M+self.M0)
        return s1, s2, p1, p2, p3, p4, CM