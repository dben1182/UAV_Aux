#%%


#this file is used to re implement the adaptive filter from DSP last year
#########################################################################


import numpy as np
import scipy as sp

import time



#gets the pyplot
import matplotlib.pyplot as plt

#imports csv so we can read in properly
import csv


#imports the least mean squares function
from least_mean_squares import least_mean_squares



########################################################################
#sets the tuning parameters for the different parts of the lms adaptive filter

#sets the mu step size scaling factor to pretty small to ensure convergence
mu = 0.001

########################################################################

#reats in each of the audio files


#reads in the z audio
with open('audio_files/other/z_audio.csv', 'r') as file:
    reader = csv.reader(file)
    z_audio = np.array(list(reader)).astype(np.float_)

#converts to 1d array
z_audio = z_audio[:,0]

print(z_audio)


#reads the p audio
with open('audio_files/other/p_audio.csv', 'r') as file:
    reader = csv.reader(file)
    eta_audio = np.array(list(reader)).astype(np.float_)

#converts to 1d array
eta_audio = eta_audio[:,0]

print(eta_audio)

#sets the sample rate to 8 kilohertz
sampleRate = 8000

#gets the length of z and p audio files
z_time = np.size(z_audio)/sampleRate
p_time = np.size(eta_audio)/sampleRate



#sets the h_initial to all zeros

#sets the length of the h filter
h_length = 120

#creates the h_initial
h_initial = np.zeros(h_length)


#passes in the p_audio, which is the actual signal, the z_audio,
#which is the desired signal, mu the stepping size, and the h_initial guess
#runs the adaptive filter and gets the final output signal y,
#the estimate for h, and the error in the signal
y, h_estimated, error = least_mean_squares(eta_audio, z_audio, mu, h_initial)


#%%

#plots the error as a function of samples
plt.figure()
plt.plot(error)
plt.title("Error as a function of samples")


#does a stem plot to compare the h_true, and h_estimated
plt.figure()
plt.stem(h_estimated)
plt.legend(['H estimated'])
plt.title("Estimated H transfer function")


#sets the number of samples of the freqz output
worN = 1024

#plots the frequency response of h_estimated transfer function
w, h = sp.signal.freqz(h_estimated, a=1, worN=worN, fs=sampleRate)


#gets the same in hertz
w_hertz = w/(2*np.pi)

#plots it out
plt.figure()
plt.title("LMS adapted filter frequency response")
plt.plot(w_hertz, np.abs(h))

#from this output, we can see that 


# %%
