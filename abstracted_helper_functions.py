#library imports
import grama as gr
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

def wec_ode(t_span,z,m,b,k,c,q,wave_interpolation):
    """
    Helper function for scipy.solve_ivp

    Args:
    t_span (list): Start and end time of the evaluation
    z,m,b,k,c,q (float): Design variables
    wave_interpolation (function): Callable function of an interpolated wave function

    Returns:
    list: List of the time series, position series and velocity series of the system response
    """
    #define the wave forcing function over the specified timespan using the interpolated wave
    wave_height = wave_interpolation(t_span)

    #define gravitational constant
    g = 9.81

    return [z[1], 1/m*(wave_height + b*g - m*g - k*z[0] - c*z[1] + q*z[1]**2 )]


def evaluate_ode(t_span,df,t_md, md_sea):
    """
    Numerically solve a harmonicly excited PA-WEC system using RK45

    Args:
    t_span (list): Start time, end time, and number of points
    df (datafram): Design variables of the system (m,b,k,c,q)
    t_md (array): Time series over which to evaluate the system response
    md_sea (array): Discretized wave function

    Returns:
    list: List of the time sereis, position series and velocity series of the system response
    """
    #unpack design variables
    m = df.m.squeeze()
    b = df.b.squeeze()
    k = df.k.squeeze()
    c = df.c.squeeze()
    q = df.q.squeeze()

    #create dataframe defining the wave function with both time and wave magnitude
    sea_conditions = gr.df_make(x=t_md.ravel(), y=md_sea.ravel())

    #interpolate discretized wave function to create callable continuous function
    wave_interpolation = interp1d(sea_conditions.x, sea_conditions.y, kind="linear")

    #pack design variables and wave function into a tuple to pass into solve_ivp
    variables = (m,b,k,c,q,wave_interpolation)
    
    #generate time series
    time = np.linspace(t_span[0],t_span[1], t_span[2])

    return solve_ivp(
        wec_ode, 
        [t_span[0], t_span[1]], 
        [0, 0], 
        method='RK45', 
        args=variables,
        t_eval = time,
        dense_output=True,
    )


##code for visualizing single wave forms

##extra imports
# import matplotlib
# matplotlib.use('TkAgg') # Or 'QtAgg' if you installed PyQt
# import matplotlib.pyplot as plt
# DF = gr.Intention()

##define wave paramters
# wave_amplitude = 0.2
# wave_frequency = 1
# wave_phase_shift = 0

##define time series
# start_time = 0
# end_time = 50
# resolution = 20
# n_points = resolution*(end_time-start_time)

##create discretized wave form
# df_sine = (
#         gr.df_make(x=np.linspace(start_time, end_time, num=n_points))
#         >> gr.tf_mutate(y=wave_amplitude*gr.sin(wave_frequency * DF.x + wave_phase_shift))
#     )

##define design variables
# m = 4454.766593
# b = 99.998951
# k = 5478.237066
# c = 1
# q = 959.341924

##pack design variables into a dataframe
# df = gr.df_make(
#     m=m,
#     b=b,
#     k=k,
#     c=c,
#     q=q,
# )

##evaluate system response
# res = evaluate_ode([start_time,end_time,n_points], df, [wave_amplitude, wave_frequency, wave_phase_shift])
#     # tmp = evaluate_ode([start_time,end_time,n_points], df, df_fit1)

##create new dataframe to assist with plotting
# ploting = gr.df_make(
#     vel = res.y[1],
#     pos = res.y[0],
#     time = res.t
#     )

##create plot of input wave, position, and velocity response
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4))

# ax1.plot(ploting.time, ploting.pos, color='blue')
# ax1.set_title('Position')

# ax2.plot(ploting.time, ploting.vel, color='red')
# ax2.set_title('Velocity')

# ax3.plot(df_sine.x, df_sine.y, color="black")
# ax3.set_title('Input Wave')

# plt.tight_layout() # Adjusts spacing to prevent overlap
# plt.show()