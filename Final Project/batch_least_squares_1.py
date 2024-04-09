#%%
#this file implements the a recursive least squares algorithm to prepare for the final project

#Here however, we implement a batch least squares all at once, which is computationally inefficient

#I am using an IIR filter and give it some input and it spits out an output
import numpy as np

#imports scipy as sp
import scipy as sp

#sets the filter order
filterOrder = 3

#sets the corner frequency
Wn = 4000

#sets the sample frequency
sampleFrequency = 10000

#gets the b and a vectors coefficients
b, a = sp.signal.iirfilter(filterOrder, Wn, btype='lowpass', fs = sampleFrequency)

#gets the original length of b
b_original_size = np.size(b)
#gets the original length of a
a_original_size = np.size(a)

#gets m, which is the size of b
m = b_original_size - 1

#gets n, which is the a original size - 1
n = a_original_size - 1


#reshapes b into column vector
b = b.reshape((b_original_size, 1))

#reshapes a into column vector
a = a.reshape((a_original_size, 1))


#shortens a to not have the first coefficient, which is not used in the calculations
a_shortened = a[1:a_original_size]

#gets the a shortened length
a_shortened_length = np.size(a_shortened)

print(a_shortened)

#concatenates a and b
coefficients = np.concatenate((a_shortened, b), axis=0)

print(coefficients)

#sets the number of coefficients
numCoefficients = np.size(coefficients)

print("num coefficients: ", numCoefficients)

#creates the signal, which will be a sine wave
signalLength = 100

#sets the sine wave frequency
frequency = 3*np.pi/17

#creates the signal
x = np.zeros((signalLength, 1))
for i in range(signalLength):
    x[i][0] = np.cos(frequency*i)


#creates the output array and initializes to zeros
y = np.zeros((signalLength, 1))

#initializes the a matrix to zeros
a_k = np.zeros((numCoefficients, 1))

#sets the mean and the variance of the noise
noise_mean = 0.0
noise_variance = 1.0

#constructs the a vector, which contains the previous output samples and the current 
#inputs samples
for k in range(signalLength):
    #resets the a array to zeros
    a_k = np.zeros((numCoefficients, 1))
    #iterates through the a coefficients
    for j in range(0, a_shortened_length):
        #checks to make sure it is within range
        if (k - j - 1) >= 0:
            #if k-j is greater than or equal to zero, than we get that sample into
            #our data vector
            a_k[j][0] = y[k-j-1]
    #iterates through the b coefficients
    for j in range(0, b_original_size):
        #checks to make sure we are within range
        if (k - j) >= 0:
            #if this is true, we extract that portion of the signal
            a_k[n+j][0] = x[k-j]
    
    #print(np.size(a))

    #now that the a vector has been constructed, we get y[k], which will be a multiplied by the a and b vector,
    #with added noise
    y[k][0] = a_k.T @ coefficients# + np.random.normal(noise_mean, noise_variance)


#print(y)





b = b.reshape(b_original_size)
a = a.reshape(a_original_size)

x = x.reshape(signalLength)

#gets the same result using other means

y_other = sp.signal.lfilter(b, a, x)


print(y_other)



# %%
