#%%


#this file is used to re implement the adaptive filter from DSP last year
#########################################################################


import numpy as np
import scipy as sp

import time

#imports the inline audio player
import sounddevice as sd


#gets the pyplot
import matplotlib.pyplot as plt

#imports csv so we can read in properly
import csv


#imports the least mean squares function
from least_mean_squares import least_mean_squares



########################################################################
#sets the tuning parameters for the different parts of the lms adaptive filter

#sets the mu step size scaling factor to pretty small to ensure convergence
mu = 0.00001


########################################################################

#reats in each of the audio files

#reats in the h_true
with open('audio_files/h_true.csv', 'r') as file:
    reader = csv.reader(file)
    h_true_string = list(reader)

h_numpy_string = np.array(h_true_string)
h_true = h_numpy_string.astype(np.float_)
print(h_true)


#reads in the z audio
with open('audio_files/z_audio.csv', 'r') as file:
    reader = csv.reader(file)
    z_audio = np.array(list(reader)).astype(np.float_)

#converts to 1d array
z_audio = z_audio[:,0]

print(z_audio)


#reads the p audio
with open('audio_files/p_audio.csv', 'r') as file:
    reader = csv.reader(file)
    p_audio = np.array(list(reader)).astype(np.float_)

#converts to 1d array
p_audio = p_audio[:,0]

print(p_audio)

#sets the sample rate to 8 kilohertz
sampleRate = 8000

#gets the length of z and p audio files
z_time = np.size(z_audio)/sampleRate
p_time = np.size(p_audio)/sampleRate

'''
#plays z and p audios 
sd.play(z_audio, sampleRate)
time.sleep(z_time)
sd.stop()

sd.play(p_audio, sampleRate)
time.sleep(z_time)
sd.stop()'''


#sets the h_initial to all zeros

#gets the h_real length
h_length = np.size(h_true)

#creates the h_initial
h_initial = np.zeros(h_length)


#passes in the p_audio, which is the actual signal, the z_audio,
#which is the desired signal, mu the stepping size, and the h_initial guess
#runs the adaptive filter and gets the final output signal y,
#the estimate for h, and the error in the signal
y, h_estimated, error = least_mean_squares(p_audio, z_audio, mu, h_initial)


#%%

#plots the error as a function of samples
plt.figure()
plt.plot(error[0:50000])


#does a stem plot to compare the h_true, and h_estimated
plt.figure()
#plt.stem(h_true)
#plt.stem(h_estimated)
plt.legend(['H true', 'H estimated'])



# %%
