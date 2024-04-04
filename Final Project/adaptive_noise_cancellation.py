#%%
#this file implements full noise cancellation

################################################################
#imports the needed libraries
import numpy as np
import scipy as sp
import time
import matplotlib.pyplot as plt
import csv
from least_mean_squares import least_mean_squares
################################################################


################################################################


#reads in eta1, eta2, z1, and z2 audio files

#reads in the eta1 audio
with open('audio_files/noise_cancel_eta.csv', 'r') as file:
    reader = csv.reader(file)
    eta_1_audio = np.array(list(reader)).astype(np.float_)

#converts to 1d array
eta_1_audio = eta_1_audio[:,0]

#reads in the eta2 audio
with open('audio_files/noise_cancel_eta2.csv;','r') as file:
    reader = csv.reader(file)
    eta_2_audio = np.array(list(reader)).astype(np.float_)    

#converts to 1d array
eta_2_audio = eta_2_audio[:,0]


#reads in the z1 audio
with open('audio_files/noise_cancel_z.csv', 'r') as file:
    reader = csv.reader(file)
    z_1_audio = np.array(list(reader)).astype(np.float_)

#converts to 1d array
z_1_audio = z_1_audio[:,0]

#reads in the z2 audio
with open('audio_files/noise_cancel_z2.csv', 'r') as file:
    reader = csv.reader(file)
    z_2_audio = np.array(list(reader)).astype(np.float_)

#converts to a 1d array
z_2_audio = z_2_audio[:,0]
################################################################


################################################################
#sets the tuning parameters of our filter

#sets the filter length
filterLength = 20

#sets the mu, or step size
mu = 0.001

#initializes the filter with zeros
h_initial = np.zeros((filterLength, 1))

################################################################


################################################################
#calls lms on the eta1 and z1

#in this case, it is a little counterintuitive. eta, or noise, will
#be our input signal (x), z will be our desired signal, and y will be an
#estimate of the noise present in the signal, which will be subtracted
#out from z, so the error will actually be our desired output signal


eta_hat, h_estimated, s_hat = least_mean_squares(eta_1_audio, z_1_audio, mu=mu, h_init=h_initial)


################################################################



# %%
