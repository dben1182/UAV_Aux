#%%
#Recursive Least Squares System Identification

#In this file, we implement a recursive least squares algorithm to prove that we can accomplish system identification

#This technology will be used in the future for system identification of an  airplane in flight, so as to
#self tune the parameters of the airplane, to increase the effectiveness of the autopilot.




#imports the needed libraries for the system to work
import numpy as  np
import matplotlib.pyplot as plt
import scipy.signal as signal
import librosa as lib
#imports the linear filter that I wrote myself
from linearFilter import linearFilter
from a_N_constructor import a_N_constructor


#-----------------------------------------------------
#Reads in a portion of a speech audio file as the signal to black box test
#the filter

x_signal, sampleRate = lib.load('Commanding_Image_Of_Christ.mp3')

#gets a shortened version of the x_signal so we don't have to process 40 miutes worth of material
shortenedLength = 100000
#we also start at another point to there is no dead period of the speech
shortenedStartPoint = 55000
x_signal_shortened = x_signal[shortenedStartPoint:(shortenedStartPoint+shortenedLength)]
#reshapes x_signal_shortened to be a column vector
x_signal_shortened = x_signal_shortened.reshape((shortenedLength, 1))


#-----------------------------------------------------


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



#---------------------------------------------------------------
#runs the linear filter on the signals with the as and bs (HA! you said 'bs')

y_signal = linearFilter(b_original, a_original, x_signal_shortened)

#adds noise to the y_signal, as will happen with all signals in the future
y_signal_noisy = np.zeros((shortenedLength, 1))

#sets the mean and variance of the signal noise
noiseMean = 0.0
noiseVariance = 0.025
#adds the noise to create the noisy signal
for n in range(shortenedLength):
    y_signal_noisy[n][0] = y_signal[n][0] + np.random.normal(loc=noiseMean, scale=noiseVariance)

#sets the lower and upper limits of the plotting
plotLower = 20000
plotUpper = 20500

plt.figure()
#plots he x_signal_shortened as well as the y signal
plt.plot(x_signal_shortened[plotLower:plotUpper])
plt.plot(y_signal[plotLower:plotUpper])
plt.plot(y_signal_noisy[plotLower:plotUpper])
plt.legend(['x', 'y', 'y with noise'])

#---------------------------------------------------------------



# %%

#-------------------------------------------------------------------------------

#The stationary Kalman filter is used to recursively update the estimate of the coefficients

#In order to effectively implement the kalman filter, we need to use the A_N constructor and get the first 10 or so samples
#of the A_N vector, and create a non-singular A.T @ A matrix, which will be our starting point for for the P matrix
# and we will take the pseudoinverse, which will be the starting point for x, or our initial estimate of x_hat

#sets the number samples to take for the initial guess
numSamplesInitial = 10

#creates the initial A matrix 
A_initialization = np.zeros((numSamplesInitial, numCoefficients))

for n in range(numSamplesInitial):
    #gets a_n for each iteration
    a_N = a_N_constructor(x_signal, y_signal_noisy, M=M, N=N, n=n)

    #adds a_N to the temporary matrix
    A_initialization[n,:] = a_N.T

    #prints the a initialization
    print(A_initialization)




# %%
