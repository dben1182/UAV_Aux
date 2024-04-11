#constructs an a_k vector based on input signals

import numpy as np



#defines constructor to create the a_N vector

#Arguments:
#1. x_signal
#2. y_signal
#3. M (last index of b vector)
#4. N (last index of a vector)
#5. n (index of y from which to construct the a_N vector)
#Returns:
#1. a_N vector
def a_N_constructor(x, y, M, N, n):
    
    #sets the number of coefficients in the a_N vector
    #The number of b coefficients is M + 1, and the number
    #of nontrivial a coefficients is N. the sum is M + 1 + N
    numCoefficients = M + 1 + N

    #initializes the a_N vector
    a_N = np.zeros((numCoefficients, 1))

    #iterates through all the a coefficients y values
    for k in range(1, N+1):
        #checks to make sure we are within bounds for y
        if n - k >= 0:
            #sets a_N
            #needs to set using k-1 because we are going from 
            #1 to N inclusive, so we need to go from 0 to N-1
            a_N[k-1][0] = -1.0*y[n-k][0]
    
    #iterates through all the b coefficients x values
    for k in range(0, M+1):
        #checks to make sure we are within bounds for x
        if n - k >= 0:
            #sets a_N
            a_N[k+N][0] = x[n-k][0]


    #returns a_N
    return a_N