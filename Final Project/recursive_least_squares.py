#%%

#in the previous file, we implemented a block or batch least squares 
#algorithm to 

import numpy as np
import matplotlib.pyplot as plt
import zplane as zp
import scipy.signal as signal
import librosa as lib
from linearFilter import linearFilter


#------------------------------------------------------------------
#this section creates the IIR Butterworth Filter

#sets the filter order
filterOrder = 3
#sets the corner Frequency
cornerFrequency = 2000
#sets the sample Frequency
sampleFrequency = 8000

#gets the b and a from the butterworth filter
b, a = signal.butter(filterOrder, Wn=cornerFrequency, btype='low', fs=sampleFrequency)

#gets the sizes of the b and a vector
b_size = np.size(b)
M = b_size - 1

#gets the size of a and N
a_size = np.size(a)
N = a_size - 1

#reshapes a and shortens it so we no longer have the trivial coefficient
a_reshaped = a[1:]
a_reshaped = a_reshaped.reshape((N,1))

#reshapes b
b_reshaped = b.reshape((b_size, 1))

#concatenates the coefficients together
coefficients = np.concatenate((a_reshaped, b_reshaped), axis=0)

#sets the number of coefficients
numCoefficients = M + N + 1

#end Filter creation
#------------------------------------------------------------------


#-------------------------------------------------------------------
#reads in an audio file
x_signal, sampleRate = lib.load('Commanding_Image_Of_Christ.mp3')

#prints the sample Rate
print("Sample Rate: ", sampleRate)

#gets the signal length
signalLengthOld = np.size(x_signal)

#sets the new signal Length, which is shortened. About four seconds equivalent
signalLengthShort = 100000
#-------------------------------------------------------------------



#-------------------------------------------------------------------
#runs the linear filter on the signals with the a and bs

y_signal = linearFilter(b, a, x_signal)



#-------------------------------------------------------------------







#%%
#section includes the Stationary Dynamics Kalman Filter

#-------------------------------------------------------------------
#filters the signal using the devised lowpass filter

#sets the noise mean and variance
noiseMean = 0.0
noiseVariance = 0.01

#creates the x_star variable, which is the estimate of x updated with each sample
#initializes x_star to all zeros
x_star = np.zeros((numCoefficients, 1))

#iterates through for each y sample

#for n in range(signalLengthShort):





#-------------------------------------------------------------------



