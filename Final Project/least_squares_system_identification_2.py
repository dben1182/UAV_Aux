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
shortenedSignalLength = 100000
#we also start at another point to there is no dead period of the speech
shortenedStartPoint = 55000
x_signal_shortened = x_signal[shortenedStartPoint:(shortenedStartPoint+shortenedSignalLength)]
#reshapes x_signal_shortened to be a column vector
x_signal_shortened = x_signal_shortened.reshape((shortenedSignalLength, 1))


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
y_signal_noisy = np.zeros((shortenedSignalLength, 1))

#sets the mean and variance of the signal noise
noiseMean = 0.0
noiseVariance = 0.025
#adds the noise to create the noisy signal
for n in range(shortenedSignalLength):
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

#noiseless sanity check


#this portion performs the initial calculations for the kalman filter
#that is, we create the initial P matrix and the initial x_star matrix
#enough samples to get a good initialization, without the significant
#computational disadvantages. (We can perform this for like a size 50
#matrix, and we just don't care.

#sets the number of initial samples from which to construct our A_initial matrix
numSamplesInitial = 10

#instantiates the initial A matrix
A_initial = np.zeros((numSamplesInitial, numCoefficients))


#creates the initial y vector
y_initial = y_signal[:numSamplesInitial,:]

#iterates through and creates the A marix
for n in range(numSamplesInitial):
    #gets the a_n for each iteration
    a_N = a_N_constructor(x_signal_shortened, y_signal, M=M, N=N, n=n)

    #adds the a_N to the A initialization matrix
    A_initial[n,:] = a_N.T

#print("A_initial: \n", A_initial)

#gets the P_N initial matrix
P_N_initial = np.linalg.inv(A_initial.T @ A_initial)

#gets the x_star initial vector
x_star_init = P_N_initial @ A_initial.T @ y_initial

#gets the initial error
initialError = x_star_init - coefficients

print("A_initial: \n", A_initial)

#prints everything out
print("x_star_init: \n", x_star_init)
print("Initial error: \n", initialError)

#now, we are going to propogate our kalman filter by one step, which 
#should be identical to simply increasing our A matrix by the next
#sample set and taking the pseudoinverse the long, normal way


#gets the a_N for the next one
a_N = a_N_constructor(x_signal_shortened, y_signal, M=M, N=N, n=numSamplesInitial)



#creates the next bigger A matrix
A = np.append(A_initial, a_N.T, axis=0)




# %%
