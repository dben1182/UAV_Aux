#%%

#in the previous file, we implemented a block or batch least squares 
#algorithm to 

import numpy as np
import matplotlib.pyplot as plt
import zplane as zp
import scipy.signal as signal
import librosa as lib
from linearFilter import linearFilter
from a_N_constructor import a_N_constructor


#------------------------------------------------------------------
#this section creates the IIR Butterworth Filter

#sets the filter order
filterOrder = 3
#sets the corner Frequency
cornerFrequency = 1000
#sets the sample Frequency
filterSampleFrequency = 22050

#gets the b and a from the butterworth filter
b, a = signal.butter(filterOrder, Wn=cornerFrequency, btype='low', fs=filterSampleFrequency)

print("b: ", b)
print("a: ", a)


#plots the frequency response of the butterworth filter
w, h = signal.freqz(b, a, worN=1024, fs=filterSampleFrequency)

plt.figure()
plt.plot(w, np.abs(h))


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
x_signal, audioSampleRate = lib.load('Commanding_Image_Of_Christ.mp3')

#prints the sample Rate
print("Sample Rate: ", audioSampleRate)

#gets the signal length
signalLengthOld = np.size(x_signal)

#sets the new signal Length, which is shortened. About four seconds equivalent
signalLengthShort = 100000

#creates the new signal
x_signal_short = x_signal[0:signalLengthShort]

#reshapes the x_signal_short
x_signal_short = x_signal_short.reshape((signalLengthShort, 1))


#-------------------------------------------------------------------



#-------------------------------------------------------------------
#runs the linear filter on the signals with the a and bs

y_signal = linearFilter(b, a, x_signal_short)

#sets the plotting limits
plotLower = 20000
plotUpper = 20500


plt.figure()
#plots the x_signal shortened
plt.plot(x_signal_short[plotLower:plotUpper])
plt.plot(y_signal[plotLower:plotUpper])



#creates a simple test signal to run the test on
test_signal = np.zeros((signalLengthShort, 1))

for i in range(signalLengthShort):
    test_signal[i][0] = np.cos((np.pi/24)*i)

testSignalOutput = linearFilter(b, a, test_signal)

#-------------------------------------------------------------------







#%%
#section includes the Stationary Dynamics Kalman Filter

#-------------------------------------------------------------------
#filters the signal using the devised lowpass filter

#sets the noise mean and variance
noiseMean = 0.0
noiseVariance = 0.01


#creates the alpha variable, which is the constant that we use to initialize the P matrix
alpha = 100000

#creates the x_star variable, which is the estimate of x updated with each sample
#initializes x_star to all zeros
x_star_recursive = np.zeros((numCoefficients, 1))

#temporarily tries to initializes the x_star vector to a closer estimate
'''x_star = np.array([[-2.0],
                   [2.0],
                   [-5.5e-01],
                   [2.0e-03],
                   [2.0e-03],
                   [2.0e-03],
                   [2.0e-03]])
'''

#creates the P matrix
P = alpha*np.eye(numCoefficients)

#creates the changing error vector
errorChange = np.zeros((signalLengthShort,1))

#creates the A matrix to contain all the a_N matricies
#we will use this to test our recursive least squares
A = np.zeros((signalLengthShort, numCoefficients))



#iterates through for each y sample to run the recursive least squares algorithm
for n in range(signalLengthShort):
    
    #gets a_N for this particular iteration
    a_N = a_N_constructor(x_signal_short, y_signal, M=M, N=N, n=n)

    #adds a_N to the main A matrix
    A[n,:] = a_N.T




    #gets temp to help with the kalman gain
    temp = a_N.T @ P @ a_N
    #gets the kalman gain
    k_n = (P @ a_N)/(1.0 + temp[0][0])

    #gets the P matrix update
    P = P - k_n @ a_N.T @ P

    #gets the x_star update
    x_star_recursive = x_star_recursive + k_n*(y_signal[n][0] - (a_N.T @ x_star_recursive)[0][0])


    #if we are at the appropriate indecies, we calculate a batch least squares
    #and compare them to the recursive least squares
    if n % 20000 == 5000:
        A_Section = A[0:n,:]
        #gets the corresponding section of y
        y_section = y_signal[0:n,:]
        #calculates the intermediate batch least squares
        x_star_intermediate = np.linalg.inv(A_Section.T @ A_Section) @ A_Section.T @ y_section

        #print("batch x star n = ", n, "\n", x_star_intermediate)
        #print("recursive x star: \n", x_star_recursive)
        print("P: \n", P)

    #I think that the problem has something to do with the negative a coefficients
    #or some problem related thereto, because the error is exceedingly small at those numbers
    #prints the k_n

    #gets the error vector
    error = x_star_recursive - coefficients

    #plots the magnitude squared of the error
    error_magnitude = (np.linalg.norm(error))**2

    errorChange[n][0] = error_magnitude


#prints the actual coefficients
print("Coefficients: \n", coefficients)

#prints the final recursive x_star
print("x star recursive: \n", x_star_recursive)

#prints the recursive x_star error
x_star_recursive_error = x_star_recursive - coefficients
print("x star recursive error: \n", x_star_recursive_error)


#gets the final batch x star
x_star_batch_final = np.linalg.inv(A.T @ A) @ A.T @ y_signal
print("x star batch final: \n", x_star_batch_final)

#gets the final batch x star error
x_star_batch_final_error = x_star_batch_final - coefficients
print("x star batch final error: \n", x_star_batch_final_error)

#plots the magnitude of the error
plt.figure()
plt.yscale("log")
plt.plot(errorChange)

#-------------------------------------------------------------------




# %%
