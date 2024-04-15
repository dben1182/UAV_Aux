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

#-------------------------------------------------------------------------------
#initialization calculations
#The stationary Kalman filter is used to recursively update the estimate of the coefficients

#In order to effectively implement the kalman filter, we need to use the A_N constructor and get the first 10 or so samples
#of the A_N vector, and create a non-singular A.T @ A matrix, which will be our starting point for for the P matrix
# and we will take the pseudoinverse, which will be the starting point for x, or our initial estimate of x_hat

#sets the number samples to take for the initial guess
numSamplesInitial = 10

#creates the initial A matrix 
A_initialization = np.zeros((numSamplesInitial, numCoefficients))


#creates the y initialization vector
y_initial = y_signal[:numSamplesInitial,:]
print("y_initial: \n", y_initial)

#does the same for the A

#iterates through and gets the A matrix
for n in range(numSamplesInitial):
    #gets a_n for each iteration
    a_N = a_N_constructor(x_signal_shortened, y_signal_noisy, M=M, N=N, n=n)

    #adds a_N to the temporary matrix
    A_initialization[n,:] = a_N.T


#gets the initial P0, which is (A_T @ A)**-1, which will serve to initialize our Kalman filter
P_N = np.linalg.inv(A_initialization.T @ A_initialization)

#gets the initial x_star, which is the batch least squares for our initial guess
x_star = P_N @ A_initialization.T @ y_initial


#prints out the P_N initial, and then prints out x_star initial
print("P_N initial: \n", P_N)
print("x_star_initial: \n", x_star)

#gets the rank of P_N to see if it is invertible here
P_N_initial_rank = np.linalg.matrix_rank(P_N)
#gets the shape of P_N
P_N_shape = np.shape(P_N)
#gets the dimensionality of P_N
P_N_dimensionality = P_N_shape[0]

print("P_N_initial_rank: \n", P_N_initial_rank)
print("P_N_shape: \n", P_N_shape)
print("P_N_dimensionality: \n", P_N_dimensionality)

#end initialization-----------------------------------------------------------------------------


#Kalman Filter (Bayesian Estimator)-------------------------------------------------------------

#for the remaining samples in the y array, we will perform the recursive kalman filter algorithm
#using the initial P_N and the initial x_star which we just calculate

#creates the A matrix
A = np.zeros((shortenedSignalLength, numCoefficients))


#iterates through the y's we haven't used yet from our initialization

for n in range(shortenedSignalLength):
    
    #gets the next a_N for this iteration
    a_N = a_N_constructor(x_signal_shortened, y_signal, M=M, N=N, n=n)

    #adds this a_N to the already existing A matrix
    A[n,:] = a_N.T

    #gets the temp to help simplify the kalman gain
    kn_helper = a_N.T @ P_N @ a_N

    #gets the kalman gain
    k_n = (P_N @ a_N)/(1.0 + kn_helper[0][0])

    #updates the P matrix
    P_N = P_N - k_n @ a_N.T @ P_N
    
    #gets the x_star_helper
    x_star_helper = a_N.T @ x_star
    #upates the x star estimate
    x_star = x_star + k_n*(y_signal[0][0] - x_star_helper[0][0])


print("x_star final: \n", x_star)
print("actual coefficients: \n", coefficients)
x_star_error = x_star - coefficients
print("x star error: \n", x_star_error)

#gets the batch error
x_star_batch = np.linalg.inv(A.T @ A) @ A.T @ y_signal
print("x star batch: \n", x_star_batch)
batch_error = x_star_batch - coefficients
print("batch error: \n", batch_error)

#-----------------------------------------------------------------------------------------------

print("A test: \n", A[8:12,:])


# %%

#tries again in another context

A_2 = np.zeros((shortenedSignalLength, numCoefficients))

for n in range(shortenedSignalLength):

    #calls the a_N constructor
    a_N = a_N_constructor(x=x_signal_shortened, y=y_signal, M=M, N=N, n=n)

    A_2[n,:] = a_N.T


#gets the new x_star
x_star_2 = np.linalg.inv(A_2.T @ A_2) @ A_2.T @ y_signal

print("A2 test: \n", A_2[8:12])

print("x star 2: \n", x_star_2)

error = x_star_2 - coefficients
print("x_star_2 error: \n", error)
# %%
