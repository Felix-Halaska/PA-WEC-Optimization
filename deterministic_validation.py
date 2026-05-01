#library imports
import grama as gr
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp, quad, trapezoid
from abstracted_helper_functions import evaluate_ode
import matplotlib
matplotlib.use('TkAgg') # Or 'QtAgg' if you installed PyQt
import matplotlib.pyplot as plt
from gaussian_wave_model import gp_fit, optimize_hyperparameter, hyperparmeter_sweep, generate_waveform
from superposition_wave_model import Distribution_manager, create_waveform
DF = gr.Intention()
pd.set_option('display.max_columns', None)

filename = input("Type filename for data to be saved under")

##BEGIN USER INPUT

#define mean wave conditions
wave_amplitude = 1
wave_frequency = [0.01,0.1,1,10]
wave_phase_shift = 0

#wave_amplitude = -wave_amplitude*wave_frequency**2

#define time to evaluate system response
start_time = 0
end_time = 10
resolution = 20
n_points = resolution*(end_time-start_time)

#choose number of restarts to avoid local minima
restarts = 10

##END USER INPUT

t_span = np.linspace(start_time, end_time, n_points).reshape(-1, 1)

results = pd.DataFrame()

for frequency in wave_frequency:

    #generate a deterministic discretized sine wave
    sine_wave = gp_fit([wave_amplitude,frequency,wave_phase_shift],[start_time,end_time,n_points]).y

    def solve (df):
            """
            Helper function to solve system response and return desired output metrics

            Args:
            df (dataframe): Contains the design variables of the system (m,b,k,c,q)

            Returns:
            dataframe: Contains the negative average system velocity and the absolute max system extension
            """
            #evaluate the system response
            tmp = evaluate_ode([start_time,end_time,n_points], df, t_span, sine_wave)

            #compute the average velocity of the system
            avg_velocity = np.mean(abs(tmp.y[1]))
            
            return gr.df_make(
                vel_min = -1*avg_velocity,
                pos_max = (max(abs(tmp.y[0]))).squeeze(),
            )
        
    #define the optimization model
    md_det_abs = (
        gr.Model()
        >> gr.cp_vec_function(
            fun=solve,
            var=["m", "b", "k", "c", "q"],
            out=["vel_min", "pos_max"],
        )
        >> gr.cp_bounds(
            #wide ranges selected to allow for system exploration
            m =(1e2, 1e4),
            b = (1e1, 1e2),
            k=(1e1, 1e4),
            c=(1e0,1e2),
            q=(1e2,1e3),
        )
    )

    #add function to enforce the minimum extension limit
    md_det_abs = (
        md_det_abs
        >> gr.cp_vec_function(
                fun = lambda df: gr.df_make(
                    min_extension = 10 - df.pos_max
                ),
                var = ["pos_max"],
                out = ["min_extension"]
            )
    )

    #add function to enforce the maximum extension limit
    md_det_abs = (
        md_det_abs
        >> gr.cp_vec_function(
                fun = lambda df: gr.df_make(
                    max_extension = df.pos_max - 10
                ),
                var = ["pos_max"],
                out = ["max_extension"]
            )
    )

    #minimize the negative average velocity of the system
    df_wec_opt = (
        md_det_abs
        >> gr.ev_min(
            out_min="vel_min",
            out_leq=["max_extension"],
            out_geq = ["min_extension"],
            n_restart=restarts,
        )
    )

    #filter out failed optimization attempts, and flip power sign
    df_wec_opt = (
        df_wec_opt
        >> gr.tf_filter(DF.success==True)
        >> gr.tf_mutate(
            vel_max = DF.vel_min * -1
        )
    )

    #find the optimal system configuration
    df_wec_max = (
        df_wec_opt
        >> gr.tf_arrange(-DF.vel_max)
        >> gr.tf_head(1)
    )

    #add optimal results to dataframe
    results = pd.concat([results, df_wec_max], ignore_index=True)

results.to_csv(filename + '.csv')

#plot system results across all realizations in addition to the mean across all realizations
fig, axs = plt.subplots(2, 3)
axs[0, 0].plot(results.m, results.vel_max)  # Top-left
axs[0,0].set_xlabel("Effective Mass")
axs[0,0].set_ylabel("Average System Velocity")
axs[0,0].legend()
axs[0,0].set_xlim(100,10000)
axs[0,0].set_ylim(0,5)

axs[0, 1].plot(results.k, results.vel_max) # Top-right
axs[0,1].set_xlabel("Effective Spring Constant")
axs[0,1].set_ylabel("Average System Velocity")
axs[0,1].legend()
axs[0,1].set_xlim(10,10000)
axs[0,1].set_ylim(0,5)

axs[1, 0].plot(results.c, results.vel_max) # Top-right
axs[1,0].set_xlabel("Effective Generator Damping")
axs[1,0].set_ylabel("Average System Velocity")
axs[1,0].legend()
axs[1,0].set_xlim(1,100)
axs[1,0].set_ylim(0,5)


axs[1, 1].plot(results.q, results.vel_max) # Top-right
axs[1,1].set_xlabel("Effective Drag")
axs[1,1].set_ylabel("Average System Velocity")
axs[1,1].legend()
axs[1,1].set_xlim(100,1000)
axs[1,1].set_ylim(0,5)


axs[1, 2].plot(results.b, results.vel_max) # Top-right
axs[1,2].set_xlabel("Effective Buoyancy")
axs[1,2].set_ylabel("Average System Velocity")
axs[1,2].legend()
axs[1,2].set_xlim(10,100)
axs[1,2].set_ylim(0,5)

plt.tight_layout()   
plt.show()