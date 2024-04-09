#%%

#second attempt at the recursive least squares


#gets the needed libraries
import numpy as np
import scipy as sp

import csv


#sets the order of the filter
filterOrder = 10

#sets the number of coefficients used in this 
numCoefficients = int(2*filterOrder + 1)

#sets the corner frequency for the lowpass filter
cornerFrequency = 3000

#sets the sample Frequency of the signal
sampleFrequency = 8000

#gets the b and a coefficients from the the iir filter function
b, a = sp.signal.iirfilter(filterOrder, cornerFrequency, btype='lowpass', fs=sampleFrequency)

#creates the sample signal x

#gets a concatenation of a and b
coefficients_with_trivial_a = np.concatenate((a, b), axis=0)
#drops the first portion for all nontrivial coefficients of a and b
coefficients_with_all_nontrivial = coefficients_with_trivial_a[1:]

coefficients_with_all_nontrivial = coefficients_with_all_nontrivial.reshape((numCoefficients, 1))

#sets the number of samples in x
numSamples = 500

x = np.zeros((numSamples, 1))

for i in range(numSamples):
    x[i][0] = np.cos((3*np.pi/16)*i) + np.cos((4*np.pi/27))


#calls the lfilter function 
y = sp.signal.lfilter(b, a, x)

#prints out y



#constructs the A matrix, to go from the filter coefficients to the output




#sets n and m
n = 10
m = 10

#instantiates the A as zeros
A = np.zeros((numSamples, numCoefficients))

#creates a_k matrix to append to A
a_k = np.zeros((1, numCoefficients))

for k in range(numSamples):

    #resets a_k to zeros
    a_k = np.zeros((1, numCoefficients))

    #iterates through and sets the portions of y
    for i in range(0,n):
        #checks to make sure we are within bounds
        if k - i - 1 >= 0:
            a_k[0][i] = y[k - i - 1][0]
    #iterates through and sets the portions of x
    for i in range(0,m+1):
        #checks to make sure we are within valid indecies
        if k - i >= 0:
            a_k[0][n+i] = x[k - i][0]

    #adds a_k to the end of A
    A[k,:] = a_k[0,:]


#rounds A to 3 decimal places
A = np.round(A, 3)
    


#Okay, so now we have A. I now need to use the pseudoinverse to find the a and b coefficients

#gets x_star using the pseudoinverse
x_star = np.linalg.inv(A.T @ A) @ A.T @ y

np.savetxt("A.csv", A, delimiter=",")

np.savetxt("X.csv", x, delimiter=",")

np.savetxt("Y.csv", y, delimiter=",")

#prints x_star, the array, and then the error
print("X_star: \n")
print(x_star)
print("Real Coefficients: \n", coefficients_with_all_nontrivial)

error = x_star - coefficients_with_all_nontrivial
print("error: \n", error)




# %%
