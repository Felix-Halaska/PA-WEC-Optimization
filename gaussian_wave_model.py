#library imports
import grama as gr
import numpy as np
import scipy as sp
import pandas as pd
from grama.fit import ft_gp, fit_gp
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.gaussian_process import GaussianProcessRegressor
DF = gr.Intention()


def gp_fit(wave_parameters,time_series):
    """
    Helper function to create a discretized sine wave based on wave parameters and defining characteristics of a time series

    Args:
    wave_parameters (list): Wave amplitude, frequency, and phase shift
    time_series (list): Start time, end time, and number of points

    Returns:
    dataframe: Contains each time step (x) and each corresponding magnitude (y)
    
    """

    #create data frame with time series, and append with magnitude values for each time step
    df_sine = (
        gr.df_make(x=np.linspace(time_series[0], time_series[1], num=time_series[2]))
        >> gr.tf_mutate(y=wave_parameters[0]*gr.sin(wave_parameters[1] * DF.x + wave_parameters[2]))
    )

    return df_sine

def hyperparmeter_sweep(discrete_function):
    """
    Helper function to test the Gaussian Process fit to a discretized function for a range of length scale parameters 
        (therefore using the RBF kernel), and summarize the results

    Args:
    discrete_function (dataframe): Defines the input function to sweep. Column of inputs titled 'x', outputs titled 'y'

    Returns:
    dataframe: The summarized mean of the hyperparameter sweep
    """
    
    #create new dataframe to record sweep results
    df_cv_all = pd.DataFrame()

    #create length scale values to evaluate
    l_all = np.logspace(-3, +1, num=31)

    #iterate through all length scale parameters
    for l in l_all:

        #perform k-fold cross-validation using a Gaussian Process
        df_cv = (
            discrete_function
            >> gr.tf_kfolds(
                k=8, 
                ft=ft_gp(var=["x"], out=["y"], kernels=RBF(l, length_scale_bounds="fixed")), 
                out=["y"],
                summaries=dict(ndme=gr.ndme),
            )
        )
        #record the results
        df_cv_all = (
            df_cv_all
            >> gr.tf_bind_rows(df_cv >> gr.tf_mutate(l=l))
        )
        
    #summarize the results
    df_cv_summary = (
        df_cv_all
        >> gr.tf_group_by("l")
        >> gr.tf_summarize(
            ndme_mu=gr.median(DF.ndme_y),
            ndme_lo=gr.quant(DF.ndme_y, p=0.25),
            ndme_hi=gr.quant(DF.ndme_y, p=0.75),
        )
    )

    return df_cv_summary

def optimize_hyperparameter(l, discrete_function):
    """
    Function to help find the optimal length scale fit for a discretized function using a Gaussian Process
        Note: Not robust for all functions due to ranges of length scales that have zero gradient

    Args:
    l (int): Length scale parameter passed by the optimizer
    discrete_function (dataframe): Defines the input function to fit. Column of inputs titled 'x', outputs titled 'y'

    Returns:
    dataframe: The summarized mean of the hyperparameter evaluation
    """
    
    #evaluate the fit of a length scale parameter using k-fold cross-validation
    df_cv = (
        discrete_function
        >> gr.tf_kfolds(
            k=4, 
            ft=ft_gp(var=["x"], out=["y"], kernels=RBF(l, length_scale_bounds="fixed")), 
            out=["y"],
            summaries=dict(ndme=gr.ndme),
            seed=101
        )
    )

    #find the mean of the results
    df_cv_summary = (
        df_cv
        #>> gr.tf_group_by("l")
        >> gr.tf_summarize(
            ndme_mu=gr.median(DF.ndme_y),
        )
    )

    return df_cv_summary.ndme_mu


def generate_waveform(l, t_span):
    """
    Helper function to generate an uncertain function over a time series using a Gaussian Process with a determined length scale 
        parameter

    Args:
    l (int): Selected length scale parameter
    t_span (list): Time series over which to generate an uncertain function

    Returns:
    array: Wave magnitude at specified time series points
    """

    #define the process based on the length scale parameter
    gp = GaussianProcessRegressor(kernel=RBF(length_scale=l))
    
    #generate a random realization
    return gp.sample_y(t_span, n_samples=1, random_state=None)