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