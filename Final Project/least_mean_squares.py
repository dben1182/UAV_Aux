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


    #n will start at the index that is the length of the h filter, and ends with the end of the signal
    for n in range(h_filter_length, signal_length):
        #flips the h vector, for convolution
        h_flipped = np.flip(h)

        #gets the section of the x signal that we will use for getting the inner product
        x_section = x[]

        #gets the inner 
         


    return y, h, error