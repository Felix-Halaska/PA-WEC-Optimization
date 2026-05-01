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
#filename structure: wave_gen_method-n_realization-n_restart-amplitude-frequency-phase_shift

##BEGIN USER INPUT

#define mean wave conditions
wave_amplitude = 0.2
wave_frequency = 10
wave_phase_shift = 0

#define time to evaluate system response
start_time = 0
end_time = 10
resolution = 20
n_points = resolution*(end_time-start_time)

#choose number of wave realizations to test
n_realization = 10

#choose number of restarts to avoid local minima
restarts = 20

#chose wave generation method "GP" for Gaussian Process, "Superposition" for superposition of multiple waves
wave_gen_method = "GP"

##END USER INPUT

#calculate acceleration amplitude
#wave_amplitude = -wave_amplitude*wave_frequency**2

#generate time series
t_span = np.linspace(start_time, end_time, n_points).reshape(-1, 1)

#check which wave generation method to use
if wave_gen_method == "GP":
    ##start GP fitting generation
    print("Using a Gaussian Process to generate random waves")

    #package wave and time parameters
    wave_parameters = [wave_amplitude,wave_frequency,wave_phase_shift]
    time_series = [start_time,end_time,n_points]

    #create discretized sine wave
    sine_wave = gp_fit(wave_parameters, time_series)

    #perform hyperparameter sweep
    df_cv_summary = hyperparmeter_sweep(sine_wave)

    #plot hyperparameter sweep results
    plt.fill_between(df_cv_summary.l, df_cv_summary.ndme_lo, df_cv_summary.ndme_hi, alpha=1/3)
    plt.plot(df_cv_summary.l, df_cv_summary.ndme_mu)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(1e-3, 1e+1)
    plt.xlabel("Hyperparameter l (-)")
    plt.ylabel("Error metric")
    plt.show()

    ##BEGIN USER INPUT

    #User chosen length scale parameter
    l = 0.029

    ##END USER INPUT
elif wave_gen_method == "Superposition":
    print("Using wave superposition to generate random waves")

    ##BEGIN USER INPUT

    #define the standard deviation of the wave's amplitude, frequency, and phase shift
    amplitude_sd = 0.2
    frequency_sd = 0.2
    phase_shift_sd = 1

    #choose the number of waves to superimpose on each other
    n_samp = 10

    ##END USER INPUT
else:
    raise ValueError("No wave generation method provided, or it is incorrect")

#define empty dataframes to store results
results = pd.DataFrame()
all_results = pd.DataFrame()

#iterate through each realization
for realization in range(n_realization):

    #check which wave generation method to use
    if wave_gen_method == "GP":
        #generate a random wave
        random_wave = generate_waveform(l, t_span)

        plt.plot(t_span, random_wave)
        plt.show

    elif wave_gen_method == "Superposition":

        md_amplitude = Distribution_manager(wave_amplitude,amplitude_sd)
        df_amplitude = md_amplitude.sample_dist(n_samp)
        df_amplitude = df_amplitude.rename(columns={'y':'amplitude'})

        md_frequency = Distribution_manager(wave_frequency,frequency_sd)
        df_frequency = md_frequency.sample_dist(n_samp)
        df_frequency = df_frequency.rename(columns={'y':'frequency'})

        md_phase_shift = Distribution_manager(wave_phase_shift,phase_shift_sd)
        df_phase_shift = md_phase_shift.sample_dist(n_samp)
        df_phase_shift = df_phase_shift.rename(columns={'y':'phase_shift'})

        df_wave_forms = pd.concat([df_amplitude, df_frequency, df_phase_shift], axis=1)

        print(df_wave_forms)
        print(t_span)

        random_wave = create_waveform(df_wave_forms, t_span, n_points) / n_samp

        print(random_wave)
        plt.plot(t_span, random_wave)
        plt.show()

    def solve (df):
        """
        Helper function to solve system response and return desired output metrics

        Args:
        df (dataframe): Contains the design variables of the system (m,b,k,c,q)

        Returns:
        dataframe: Contains the negative average system velocity and the absolute max system extension
        """
        #evaluate the system response
        tmp = evaluate_ode([start_time,end_time,n_points], df, t_span, random_wave)

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
    
    ##plot all results for the wave realization
    # fig, axs = plt.subplots(2, 3)
    # axs[0, 0].scatter(df_wec_opt.m, df_wec_opt.vel_max)  # Top-left
    # axs[0,0].scatter(df_wec_max.m, df_wec_max.vel_max, label="Maximum")
    # axs[0,0].set_xlabel("Effective Mass")
    # axs[0,0].set_ylabel("Average System Velocity")

    # axs[0, 1].scatter(df_wec_opt.k, df_wec_opt.vel_max) # Top-right
    # axs[0,1].scatter(df_wec_max.k, df_wec_max.vel_max, label="Maximum")
    # axs[0,1].set_xlabel("Effective Spring Constant")
    # axs[0,1].set_ylabel("Average System Velocity")

    # axs[1, 0].scatter(df_wec_opt.c, df_wec_opt.vel_max) # Top-right
    # axs[1,0].scatter(df_wec_max.c, df_wec_max.vel_max, label="Maximum")
    # axs[1,0].set_xlabel("Effective Generator Damping")
    # axs[1,0].set_ylabel("Average System Velocity")

    # axs[1, 1].scatter(df_wec_opt.q, df_wec_opt.vel_max) # Top-right
    # axs[1,1].scatter(df_wec_max.q, df_wec_max.vel_max, label="Maximum")
    # axs[1,1].set_xlabel("Effective Drag")
    # axs[1,1].set_ylabel("Average System Velocity")

    # axs[1, 2].scatter(df_wec_opt.b, df_wec_opt.vel_max) # Top-right
    # axs[1,2].scatter(df_wec_max.b, df_wec_max.vel_max, label="Maximum")
    # axs[1,2].set_xlabel("Effective Buoyancy")
    # axs[1,2].set_ylabel("Average System Velocity")
    # plt.tight_layout()   
    # plt.show()


#find the average results across all realization
#average_results = results.mean()

results.to_csv(filename + '.csv')

#plot system results across all realizations in addition to the mean across all realizations
fig, axs = plt.subplots(2, 3)
axs[0, 0].scatter(results.m, results.vel_max)  # Top-left
#axs[0,0].scatter(average_results.m, average_results.vel_max, label="Mean")
axs[0,0].set_xlabel("Effective Mass")
axs[0,0].set_ylabel("Average System Velocity")
axs[0,0].legend()
axs[0,0].set_xlim(100,10000)
axs[0,0].set_ylim(0,5)

axs[0, 1].scatter(results.k, results.vel_max) # Top-right
#axs[0,1].scatter(average_results.k, average_results.vel_max, label="Mean")
axs[0,1].set_xlabel("Effective Spring Constant")
axs[0,1].set_ylabel("Average System Velocity")
axs[0,1].legend()
axs[0,1].set_xlim(10,10000)
axs[0,1].set_ylim(0,5)

axs[1, 0].scatter(results.c, results.vel_max) # Top-right
#axs[1,0].scatter(average_results.c, average_results.vel_max, label="Mean")
axs[1,0].set_xlabel("Effective Generator Damping")
axs[1,0].set_ylabel("Average System Velocity")
axs[1,0].legend()
axs[1,0].set_xlim(1,100)
axs[1,0].set_ylim(0,5)


axs[1, 1].scatter(results.q, results.vel_max) # Top-right
#axs[1,1].scatter(average_results.q, average_results.vel_max, label="Mean")
axs[1,1].set_xlabel("Effective Drag")
axs[1,1].set_ylabel("Average System Velocity")
axs[1,1].legend()
axs[1,1].set_xlim(100,1000)
axs[1,1].set_ylim(0,5)


axs[1, 2].scatter(results.b, results.vel_max) # Top-right
#axs[1,2].scatter(average_results.b, average_results.vel_max, label="Mean")
axs[1,2].set_xlabel("Effective Buoyancy")
axs[1,2].set_ylabel("Average System Velocity")
axs[1,2].legend()
axs[1,2].set_xlim(10,100)
axs[1,2].set_ylim(0,5)

plt.tight_layout()   
plt.show()