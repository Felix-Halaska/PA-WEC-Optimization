#library imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('TkAgg') # Or 'QtAgg' if you installed PyQt
import matplotlib.pyplot as plt

#assumed solution constants
C = 0.1
alpha = 0.6

#define time scale
start_time = 0
end_time = 20
resolution = 10
n_points = resolution * (end_time - start_time)

t_span = np.linspace(start_time, end_time, n_points)

def test_ode(t,z,m_0,b_0,k_0,c_0,q_0):
    """
    Helper function for scipy.solve_ivp

    Args:
    t (list): Start and end time of the evaluation
    z (list): System distance and velocity
    m_0,b_0,k_0,c_0,q_0 (float): Design variables

    Returns:
    list: List of the time series, position series and velocity series of the system response
    """
    #define the gravitational constant
    g = 9.81

    #define the assumed solution and its derivatives
    x = C * np.exp(alpha*t)
    x_dot = C * alpha * np.exp(alpha*t)
    x_d_dot = C * alpha**2 * np.exp(alpha*t)

    #define the forcing function over the specified timespan
    forcing_func = m_0 * x_d_dot - q_0 * x_dot*abs(x_dot) + c_0 * x_dot + k_0 * x + b_0

    return [z[1], 1/m_0*(forcing_func + b_0*g - m_0*g - k_0*z[0] - c_0*z[1] + q_0*z[1]*abs(z[1]) )]


def evaluate_test(t_span):
    """
    Numerically solve the PA-WEC system using RK45 for MMS verification

    Args:
    t_span (list): Start time, end time, and number of points

    Returns:
    list: List of the time sereis, position series and velocity series of the system response
    """
    #define design variables
    m_0 = 5000
    b_0 = 50
    c_0 = 10
    k_0 = 10000
    q_0 = 10

    #pack design variables and wave function into a tuple to pass into solve_ivp
    variables = (m_0,b_0,k_0,c_0,q_0)
    
    #generate time series
    time = np.linspace(t_span[0],t_span[1], t_span[2])

    return solve_ivp(
        test_ode, 
        [t_span[0], t_span[1]], 
        [0, 0], 
        method='RK45', 
        args=variables,
        t_eval = time,
        dense_output=True,
    )

#numerically calculate the solution to the system
MMS_sol = evaluate_test([start_time,end_time,n_points])

#redefine the assumed solution
assumed_sol = C * np.exp(alpha*t_span)

#plot the assumed solution against the calculated solution
plt.plot(t_span,assumed_sol, label="Assumed Solution")
plt.plot(MMS_sol.t,MMS_sol.y[0], label="Calculated Solution")
plt.xlabel("Time")
plt.ylabel("Displacement")
plt.legend()
plt.show()

