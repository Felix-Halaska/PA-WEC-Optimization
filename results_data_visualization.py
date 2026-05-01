import pandas as pd
import numpy as np
import grama as gr
import matplotlib
matplotlib.use('TkAgg') # Or 'QtAgg' if you installed PyQt
import matplotlib.pyplot as plt
DF = gr.Intention()

#import results csv files
results = pd.read_csv('Frequency-Parameter-Sweep-NewModel-2.csv')

#calculate the natural frequency for each frequency response
results = (
    results
    >> gr.tf_mutate(
        wn = (DF.k / DF.m) ** 0.5
    )
)

#list frequencies present in the results, in order
frequencies = [0.01,1,10]

#plot system results across all realizations in addition to the mean across all realizations
fig, axs = plt.subplots(2, 3)
axs[0, 0].scatter(frequencies, results.m)  # Top-left
axs[0,0].set_xlabel("Wave Frequency")
axs[0,0].set_ylabel("Effective Mass")
axs[0,0].set_xscale('log')

axs[0, 1].scatter(frequencies,results.k) # Top-right
axs[0,1].set_xlabel("Wave Frequency")
axs[0,1].set_ylabel("Effective Spring Constant")
axs[0,1].set_xscale('log')

axs[1, 0].scatter(frequencies,results.c) # Top-right
axs[1,0].set_xlabel("Wave Frequency")
axs[1,0].set_ylabel("Effective Generator Damping")
axs[1,0].set_xscale('log')


axs[1, 1].scatter(frequencies,results.q) # Top-right
axs[1,1].set_xlabel("Wave Frequency")
axs[1,1].set_ylabel("Effective Drag")
axs[1,1].set_xscale('log')


axs[1, 2].scatter(frequencies, results.b) # Top-right
axs[1,2].set_xlabel("Wave Frequency")
axs[1,2].set_ylabel("Effective Buoyancy")
axs[1,2].set_xscale('log')

axs[0,2].scatter(frequencies, results.wn)
axs[0,2].set_xlabel("Wave Frequency")
axs[0,2].set_ylabel("Natural Frequency")
axs[0,2].set_xscale('log')

plt.tight_layout()   
plt.show()
