#This file implements a basic IIR Filter for a given input signal
import numpy as np


#both input x and output y are 2d column vectors


#takes as an argument the b coefficients, which define the zeros
#and the a coefficients which define the poles of the system
#and the signal itself
def linearFilter(b, a, x):

    #gets M and N from b and a respectively
    M = np.size(b) - 1
    N = np.size(a) - 1


    #gets the signal length
    signalLength = np.size(x)

    #initializes the y output vector
    y = np.zeros((signalLength,1))

    #iterates through every element in the input signal
    for n in range(signalLength):
        
        #sets the sum for the feedback portion of the filter
        a_sum = 0.0
        #iterates through each previous output sample and corresponding a coefficient
        for k in range(1,N+1):
            
            #checks to make sure we are indexing within bounds
            if n - k >= 0:

                #adds the feedback terms to the sum
                a_sum = a_sum + a[k]*y[n-k][0]


        #sets the sum for the input portion of the filter
        b_sum = 0.0
    
        #iterates through each previous input sample in the range
        for k in range(0, M+1):
            #checks to make sure that the x sample is withing indexing bounds
            if n - k >= 0:
                #adds the b sum together
                b_sum = b_sum + b[k]*x[n-k][0]
        #sets the y output samples

        y[n][0] = b_sum - a_sum

    #returns the y signal
    return y


