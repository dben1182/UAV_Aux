#%%
#imports the needed libraries for the system to work
import numpy as  np
import matplotlib.pyplot as plt
import scipy.signal as signal
import librosa as lib
#imports the linear filter that I wrote myself
from linearFilter import linearFilter
from a_N_constructor import a_N_constructor

from parameterEstimators import stationaryKalmanFilter


#-----------------------------------------------------
#Reads in a portion of a speech audio file as the signal to black box test
#the filter

x_signal, sampleRate = lib.load('Commanding_Image_Of_Christ.mp3')

#gets a shortened version of the x_signal so we don't have to process 40 miutes worth of material
shortenedSignalLength = 100000
#we also start at another point to there is no dead period of the speech
shortenedStartPoint = 55000
x_signal_shortened = x_signal[shortenedStartPoint:(shortenedStartPoint+shortenedSignalLength)]
#reshapes x_signal_shortened to be a column vector
x_signal_shortened = x_signal_shortened.reshape((shortenedSignalLength, 1))


#-------------------------------------------------------------
#In order to work up to airplane system identification, we need to work with a super super simple system, which
#in this case is a third order butterworth lowpass IIR filter. 

#sets the filter order
filterOrder = 3
#sets the corner frequency
cornerFrequency = 1000

#gets the filter
b_original, a_original = signal.butter(filterOrder, Wn=cornerFrequency, btype='low', fs=sampleRate)

#sets the a size and N
a_size = np.size(a_original)
N = a_size - 1

#sets the b size and M
b_size = np.size(b_original)
M = b_size - 1


#reshapes a and b to be 2d column vectors and then stacks them

#reshapes a to omit the first (trivial) sample
a_reshaped = a_original[1:]
#reshapes a to be a column vector
a_reshaped = a_reshaped.reshape((N, 1))

#reshapes b to be a column vector
b_reshaped = b_original.reshape((b_size,1))


#creates the stacked coefficients
coefficients = np.concatenate((a_reshaped, b_reshaped), axis=0)

#sets the number of coefficients
numCoefficients = N + M + 1

#---------------------------------------------------------------
#runs the linear filter on the signals with the as and bs

y_signal = linearFilter(b_original, a_original, x_signal_shortened)

#sets the time between samples
Ts = 1/sampleRate

#sets the initial x_star_init
x_star_init = np.zeros((numCoefficients, 1))

#sets the P init
P_init = np.zeros((numCoefficients, numCoefficients))

#initalizes the kalman filter
kalmanFilter = stationaryKalmanFilter(ts=Ts, x_init=x_star_init, P_init=P_init, y_length=1)


#iterates through to do system characterization using the kalman filter
for n in range(shortenedSignalLength):
    #gets the a_N
    a_N = a_N_constructor(x_signal_shortened, y_signal, M=M, N=N, n=n)
    
    #gets the scalar y[n] and then reshapes it as a 2d array
    y_n_scalar = y_signal[n,0]
    #reshapes it as a 2d array so that it 
    y_n = np.reshape(y_n_scalar, (1,1))
    #calls the update function for the kalman filter
    kalmanFilter.update(y_n, a_N)


print("x_star final: \n", kalmanFilter.x_star)

#gets the x star error
x_star_error = kalmanFilter.x_star - coefficients
print("x star error: \n", x_star_error)
    
# %%
