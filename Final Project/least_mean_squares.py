#This is the least mean squares function, which implements a least
#means squares filter

import numpy as np

#this file takes as an input the x or input signal, the desired signal
#the mu, which is the step size, and the h_init, which is the initial h
#vector which is the adaptive FIR filter
def least_mean_squares(x, desired_signal, mu, h_init):

    #gets the length of the h filter
    h_filter_length = np.size(h_init)

    #gets the signal length
    signal_length = np.size(x)

    #sets the current h to the initial h, though this will change as time goes on
    h = h_init

    #creates the error vector, which stores the error between y and the
    #desired signal
    error = np.zeros((signal_length, 1))

    #creates an output signal vector, y
    y = np.zeros((signal_length, 1))

    #iterates the function through a specified number of times of propogation
    #in order to get. We iterate through the samples after the ones not fully affected
    #before the convolution has fully taken place


    #n will start at the index that is the length of the h filter, and ends with a buffer of length h filter at the end
    for n in range(0, signal_length-h_filter_length):
        #print("H size: ", np.size(h))
        #gets the section of the x signal that we will use for getting the inner product
        #the length of the section will be the length of the h filter
        x_section = x[n:n + h_filter_length]

        #gets the x_section_reverse_ordered
        x_section_reverse_ordered = np.flip(x_section)

        
        #print("x_section size: ", np.size(x_section))
        #gets the inner product between the he flipped and the x_section
        #in order to get the next value for y
        y[n] = np.inner(h, x_section)

        #using the desired signal, we get the error between the desired and actual signal
        error[n] = desired_signal[n] - y[n]

        #uses gradient descent in order to create a better h filter closer to the optimal h
        #mu is a step size scaling factor. error(n) sets the scaling factor as well, which is
        #adaptive. the larger the error, the larger the step. The smaller the error, the smaller
        #the step. And adds the very section that we just used from the signal itself. In this case,
        #we need to flip this section of x to get proper convergence
        h = h + mu*x_section_reverse_ordered*error[n]


         

    #returns the y, h, and 
    return y, h, error