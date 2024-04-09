#%%

#This file is my third attempt to characterize an IIR System using the outputs with added noise
import numpy as np
import matplotlib.pyplot as plt
import zplane as zp
import scipy.signal as signal


#sets the filter order to a low number
filterOrder = 3

#sets the corner frequency
cornerFrequency = 3000

#sets the sampleFrequency
sampleFreqency = 8000

#gets a butterworth filter, and the corresponding b and a coefficients
b, a = signal.butter(filterOrder, Wn=cornerFrequency, btype='low', fs=sampleFreqency)

print("b: \n", b)
print("a: \n", a)

#gets M, which is the last index of b and the length of b
b_size = np.size(b)
M = b_size - 1

#gets N, which is the last index of a and the length of a
a_size = np.size(a)
N = a_size - 1

#gets the transfer function of the butterworth filter
transferFunction = signal.TransferFunction(b, a)
print("Transfer Function: ", transferFunction)

#gets the normalized frequency response
zp.freq(transferFunction, type='solid', grid=True, name='Order 3 Lowpass Filter')


#gets the Pole-Zero plot
zp.pz(transferFunction)


#sets the signal length
signalLength = 700

#creates a signal for the matrix
x_signal = np.zeros((signalLength, 1))

for i in range(signalLength):
    x_signal[i][0] = np.cos((np.pi/17)*i + np.pi/2)


#uses lfilter to create the y array
y_signal = signal.lfilter(b, a, x_signal)



#creates a new y signal
y_signal_new = np.zeros((signalLength, 1))

#attempts to perform the summation to create each output sample
for n in range(signalLength):
    #iterates through each of a coefficients
    
    #creates the a sum variable
    a_sum = 0.0
    #iterates through the previous output signal samples
    for k in range(1,N+1):
        #checks to make sure that the y sample is within range
        if n - k >= 0:
            #adds the feedback sums
            a_sum = a_sum + a[k]*y_signal_new[n-k][0]


    #creates the b sum variable
    b_sum = 0.0
    #iterates through the input signal
    for k in range(0,M+1):
        #checks to make sure that the x sample is within range
        if n - k >= 0:
            #adds the input signal sums
            b_sum = b_sum + b[k]*x_signal[n-k][0]

    #sets the y output signal to the sum
    y_signal_new[n][0] = -a_sum + b_sum


y_signal_sanity_check = np.zeros((signalLength, 1))
#creates sanity check y
for n in range(10,signalLength):
    a_sum = 0.0
    for k in range(1,N+1):
        a_sum = a_sum + a[k]*y_signal_sanity_check[n-k][0]
    
    b_sum = 0.0
    for k in range(0, M+1):
        b_sum = b_sum + b[k]*x_signal[n-k][0]
    
    y_signal_sanity_check[n][0] = -a_sum + b_sum

#gets the error signal between the y_signal and the y_signal_new

y_error = y_signal_new - y_signal

print("Max Error: ", np.max(y_error))

#prints out the error over various positional indecies

print("Y signal: ", y_signal[40:45])
print("Y signal new: ", y_signal_new[40:45])
print("Y Error: ", y_error[40:45])

plt.figure()

plt.plot(y_signal[0:100])
plt.plot(y_signal_new[0:100])
plt.plot(y_signal_sanity_check[0:100])




# %%
