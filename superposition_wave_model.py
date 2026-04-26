#library imports
import grama as gr
import numpy as np
import scipy as sp
import matplotlib
matplotlib.use('TkAgg') # Or 'QtAgg' if you installed PyQt
import matplotlib.pyplot as plt
import pandas as pd
DF = gr.Intention()
from grama.models import make_prlc_rand


class Distribution_manager:
    """
    A class for creating and sampling Gaussian distributions

    Attributes:
    md.normal (model): A PyGrama model that contains a Gaussian distribution
    df_data (dataframe): A dataframe containing a number of samples from a Gaussian distribution
    """
    def __init__(self,mean,sd):
        """
        Initialize a Gaussian distribution

        Args:
        mean (float): The mean of the Gaussian distribution
        sd (float): The standard deviation of the Gaussian distribution
        """
        #create a Gaussian distribution with no dependence
        self.md_normal = (
            gr.Model()
            >> gr.cp_marginals(
                y=gr.marg_mom("norm", mean=mean, sd=sd)
            )
            >> gr.cp_copula_independence()
        )

    def sample_dist(self,n):
        """
        A helper function to sample a created Gaussian distribution model

        Args:
        n (int): The number of samples to be drawn from the Gaussian distribution

        Returns:
        dataframe: Contains the samples drawn
        """
        #sample the distributions n times
        df_data = (
            self.md_normal
            >> gr.ev_sample(n=n, df_det="nom", skip=True)
            # >> gr.tf_mutate(id=DF.index)
        )
        return df_data
    
def create_waveform(df, t_span, n_points):
    """
    A helper function to create a single discretized function out of multiple sine functions defined by their amplitude,
        frequency, and phase shift

    Args:
    df (dataframe): With columns defining the amplitude, wavelength, and phase shift for a number of periodic functions
    t_span (array): An array of a time series
    n_points (int): The resolution of the final discretized function

    Returns:
    dataframe: Defines the magnitude of the new wave at the resolution of n_points
    """

    #create empty array
    total_wave = np.empty((n_points,1))

    #iterate through all sine functions and combine
    for row in df.itertuples():
        total_wave =  total_wave + row.amplitude*np.sin(row.frequency*t_span + row.phase_shift)
        
    #select magnitude
    total_wave = total_wave[0]

    return total_wave



## Testing code for plotting single waves

# start_time = 0
# end_time = 10
# resolution = 20
# n_points = resolution*(end_time-start_time)

# t_span = np.linspace(start_time, end_time, n_points)

# df_sine = (
#         gr.df_make(x=np.linspace(start_time, end_time, num=n_points))
#         >> gr.tf_mutate(y=0.2*gr.sin(1 * DF.x + 0))
#     )

# n_samp = 10

# md_amplitude = Distribution_manager(0.2,0.1)
# df_amplitude = md_amplitude.sample_dist(n_samp)
# df_amplitude = df_amplitude.rename(columns={'y':'amplitude'})

# md_frequency = Distribution_manager(1,0.1)
# df_frequency = md_frequency.sample_dist(n_samp)
# df_frequency = df_frequency.rename(columns={'y':'frequency'})

# md_phase_shift = Distribution_manager(0,0.1)
# df_phase_shift = md_phase_shift.sample_dist(n_samp)
# df_phase_shift = df_phase_shift.rename(columns={'y':'phase_shift'})

# df_wave_forms = pd.concat([df_amplitude, df_frequency, df_phase_shift], axis=1)

# # print(df_amplitude)
# # print(df_frequency)
# # print(df_phase_shift)
# # print(df_wave_forms)

# final_form = create_waveform(df_wave_forms, t_span, n_points) / n_samp
# final_form = final_form[0]
# # final_form = np.transpose(final_form)

# print(type(final_form))
# print(final_form)
# print(final_form.shape)

# # plt.plot(t_span, final_form)
# # plt.scatter(df_sine.x, df_sine.y, color="black")
# # plt.xlabel("Time")
# # plt.ylabel("Wave Height")
# # plt.show()

# # plt.scatter(amplitude_sample.id, amplitude_sample.amplitude)
# # plt.show()
    
