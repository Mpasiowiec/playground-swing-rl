def RK4_for_2nd_order_ODE(fun, h, t, x, x_dot):
    """
    Perform one step of the classical 4th-order Runge-Kutta (RK4) integration method
    for a second-order ordinary differential equation (ODE) of the form:
        x'' = fun(t, x, x_dot),
    where x is the position and x_dot is the velocity.

    The second-order ODE is integrated by treating it as a system of two first-order ODEs:
        x' = x_dot,
        x_dot' = fun(t, x, x_dot),

    Parameters:
    - fun: callable function representing the acceleration, i.e. x'' = fun(t, x, x_dot)
    - h: timestep size
    - t: current time
    - x: current position
    - x_dot: current velocity

    Returns:
    - x_next: estimated position after timestep h
    - x_dot_next: estimated velocity after timestep h

    Reference:
    This implementation is based on the classical RK4 method adapted for second-order ODEs.
    See https://math.stackexchange.com/questions/2615672/solve-fourth-order-ode-using-fourth-order-runge-kutta-method for detailed explanation.

    Note:
    The code carefully computes intermediate slopes for both position and velocity,
    then combines them with RK4 weights to produce the next step.
    """
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