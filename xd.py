from math import sin, cos, sqrt, pi
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def RK4_for_2nd_order_ODE(fun, h, t, x, x_dot):
    # RK4 for 2nd order ODE https://math.stackexchange.com/questions/2615672/solve-fourth-order-ode-using-fourth-order-runge-kutta-method
    # in picture person forgot to change notation in last v in formulas for dvs
    
    dx1 = h*x_dot
    dx_dot1 = h*fun(t, x, x_dot)
    
    dx2 = h*(x_dot+dx_dot1/2)
    dx_dot2 = h*fun(t+h/2, x+dx1/2, x_dot+dx_dot1/2)
    
    dx3 = h*(x_dot+dx_dot2/2)
    dx_dot3 = h*fun(t+h/2, x+dx2/2, x_dot+dx_dot2/2)
    
    dx4 = h*(x_dot+dx_dot3)
    dx_dot4 = h*fun(t+h, x+dx3, x_dot+dx_dot3)
    
    dx = (dx1+2*dx2+2*dx3+dx4)/6
    dx_dot = (dx_dot1+2*dx_dot2+2*dx_dot3+dx_dot4)/6
    
    return x+dx, x_dot+dx_dot

g = 9.8
t_step = 1/30

# drag coefficient
k = 0.4
k_prime = 7/3

# length and mass of swing
# L = 1.61
L = 1.81
# L = 2.01
M0 = 2 # base on google answer

# mass of swinger
m1 = 31.3
m2 = 12.3
m3 = 6.4
M = m1 + m2 + m3
# height of swinger
l1 = 0.792
l2 = 0.393
l3 = 0.395
# position of swinger on sit
a = (m2/2 + m3)*l2/M
b = (m1 + m2/2)*l2/M

# range of torso position
phi_mean = np.radians(10)
phi0 = np.radians(30)
# range of legs position (disregarded in paper because of significantly smaller than that of the upper body)
psi_mean = np.radians(10)
psi0 = np.radians(45)

# Moments of inertia
I = (M0/3 + M) * L**2
I_prime = (m1*l1**2 + m2*l2**2 + m3*l3**2)/3 + (m1*m3 - m2**2/4)*l2**2/M

N = (M0/2 + M)*L

# position of swing
theta = np.radians(-33)
theta_dot = 0
# theta_ddot
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
    

title = 'my sim'
swinger_strategy = 'papers-model'
swings = 10

# initial phase of swinger
if swinger_strategy == 'ffm':
      alpha = np.radians(-20)
elif swinger_strategy == 'square-wave':
      alpha = np.radians(0)
elif swinger_strategy == 'papers-model':
      alpha = np.radians(65)
else: # motionless swinger
      alpha = np.radians(0)

# fig, ax = plt.subplots(figsize=(6.4,3.6), dpi=300)
# ax.set_aspect('equal')
# ax.set_xlim([-1.5,1.5])
# ax.set_ylim([-2.5,0])
# ax.set_xlabel('x (m)')
# ax.set_ylabel('y (m)')
# ax.set_title(title, fontsize='medium')
# fig.tight_layout()

# drawing theta on plot with its value in degrees
def plot_theta(theta):
    ax.plot([1*sin(theta),1.1*sin(theta)], [-1*cos(theta),-1.1*cos(theta)], color='black', linewidth=1)
    ax.text(1.2*sin(theta), -1.2*cos(theta), f'${np.degrees(theta):.0f}\\degree$', fontsize=8, horizontalalignment="center", verticalalignment="center")

def plot_positions():
    # position of the butt
    p1 = np.array((L*sin(theta) - a*cos(theta), -L*cos(theta) - a*sin(theta)))
    # position of knees
    p2 = np.array((L*sin(theta) + b*cos(theta), -L*cos(theta) + b*sin(theta)))
    # position of head
    p3 = p1 + l1*np.array((-sin(theta+phi), cos(theta+phi)))
    # position of feet
    p4 = p2 + l3*np.array((sin(theta+psi), -cos(theta+psi)))
        
    plot0,=ax.plot([0,L*sin(theta)],[0,-L*cos(theta)], color='grey', linewidth=0.5)
    plot1,=ax.plot([p1[0],p2[0]],[p1[1],p2[1]], color='black', linewidth=1.5)
    plot2,=ax.plot([p1[0],p3[0]],[p1[1],p3[1]], color='black', linewidth=1.5)
    plot3,=ax.plot([p2[0],p4[0]],[p2[1],p4[1]], color='black', linewidth=1.5)
    
    # center of the mass
    mc1 = (p1+p3)/2
    mc2 = (p1+p2)/2
    mc3 = (p2+p4)/2
    CM = (m1*mc1 + m2*mc2 + m3*mc3 + M0*np.array((L/2*sin(theta), -L/2*cos(theta))))/(M+M0)
    plot4,=ax.plot([CM[0]],[CM[1]],marker='o',markersize=1,color='red')
    plot5,=ax.plot([mc1[0]],[mc1[1]],marker='o',markersize=1,color='pink')
    plot6,=ax.plot([mc3[0]],[mc3[1]],marker='o',markersize=1,color='pink')
i = 0   
for i in range(swings):
      # the amplitude of the swing at the back extreme of the cycle - cycle is counted from back extreme
      phi,psi = 0, 0
      A = abs(theta)
      I_p = m1*l1*(-L*cos(phi) + a*sin(phi)) \
            +m3*l3*(L*cos(psi) + b*sin(psi))
      C_s = N - m1/2*l1*cos(phi) + m3/2*l3*cos(psi)
      C_c = -m1/2*l1*sin(phi) + m3/2*l3*sin(psi)
      # period of n-th cycle of the swing
      # added 0.95 to minimizes difference between expected period and actual
      T_n = 0.95*4*sp.special.ellipk(sin(A/2))*sqrt((I+I_prime+I_p)/(g*sqrt(C_s**2+C_c**2)))
      # frequency of swinger
      omega = 2*pi/T_n
      omega_const = 2.39 # from perplexity

      t, theta_dot, swingdone = 0, 0, False
      while True:
            if swinger_strategy == 'ffm':
                  phi = phi0*sin(omega_const*t-alpha)
                  psi = psi0*sin(omega_const*t-alpha)
                  phi_dot = phi0*omega_const*cos(omega_const*t-alpha)
                  psi_dot = psi0*omega_const*cos(omega_const*t-alpha)
                  phi_ddot = -phi0*omega_const**2*sin(omega_const*t-alpha)
                  psi_ddot = -psi0*omega_const**2*sin(omega_const*t-alpha)
            elif swinger_strategy == 'square-wave':
                  phi = phi_mean+phi0 if not swingdone else phi_mean
                  psi = psi_mean+psi0 if not swingdone else psi_mean
                  phi_dot = 0
                  psi_dot = 0
                  phi_ddot = 0
                  psi_ddot = 0
            elif swinger_strategy == 'papers-model':
                  phi = phi0*cos(omega*t-alpha)+phi_mean
                  psi = psi0*cos(omega*t-alpha)+psi_mean
                  phi_dot = -phi0*omega*sin(omega*t-alpha)
                  psi_dot = -psi0*omega*sin(omega*t-alpha)
                  phi_ddot = -phi0*omega**2*cos(omega*t-alpha)
                  psi_ddot = -psi0*omega**2*cos(omega*t-alpha)
            else: # motionless swinger
                  phi = phi_mean
                  psi = psi_mean
                  phi_dot = 0
                  psi_dot = 0
                  phi_ddot = 0
                  psi_ddot = 0
            
            if i % 2 == 0:
              # position of the swingers butt
              p1 = np.array((L*sin(theta)-a*cos(theta), -L*cos(theta)-a*sin(theta)))
              # position of the swingers knees
              p2 = np.array((L*sin(theta)+b*cos(theta), -L*cos(theta)+b*sin(theta)))
              # position of the swingers head
              p3 = p1 + l1*np.array((-sin(theta+phi), cos(theta+phi)))
              # position of the swingers feet
              p4 = p2 + l3*np.array((sin(theta+psi), -cos(theta+psi)))
            #   plot0,=ax.plot([0,L*sin(theta)],[0,-L*cos(theta)],color='black')
            #   plot1,=ax.plot([p1[0],p2[0]],[p1[1],p2[1]],color='red')
            #   plot2,=ax.plot([p1[0],p3[0]],[p1[1],p3[1]],color='green')
            #   plot3,=ax.plot([p2[0],p4[0]],[p2[1],p4[1]],color='blue')
            if i % 2 == 0:
              mid1 = (p1+p3)/2
              mid2 = (p1+p2)/2
              mid3 = (p2+p4)/2
              cm = (m1*mid1 + m2*mid2 + m3*mid3 + M0*np.array((L/2*sin(theta), -L/2*cos(theta))))/(M+M0)
            #   plot4,=ax.plot([cm[0]],[cm[1]],marker='o',markersize=1,color='red')
            # if i % 2 == 0:
            #   plt.draw()
            #   plt.pause(0.001)
            #   plot0.remove()
            #   plot1.remove()
            #   plot2.remove()
            #   plot3.remove()
            
            theta, theta_dot = RK4_for_2nd_order_ODE(eq_theta_ddot, t_step, t, theta, theta_dot)
            
            t += t_step
            i += 1
             
            if not swingdone and theta_dot < 0:
              # right extent of swing
            #   plot_theta(theta)
              swingdone = True
            if swingdone and theta < 0 and theta_dot > 0:
              # left extent of swing
            #   plot_theta(theta)
              print("period: expected=%f actual=%f" % (T_n, t))
              break
print("theta: %f" % np.degrees(theta))
