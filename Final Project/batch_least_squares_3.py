#%%

#This file is my third attempt to characterize an IIR System using the outputs with added noise
import numpy as np
import matplotlib.pyplot as plt
import zplane as zp
import scipy.signal as signal


#sets the filter order to a low number
filterOrder = 3

#sets the corner frequency
cornerFrequency = 300

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


#concatenates everything for the coefficients
a_reshaped = a[1:]
a_reshaped = a_reshaped.reshape((N,1))

b_reshaped = b.reshape((b_size,1))

coefficients = np.concatenate((a_reshaped, b_reshaped), axis=0)


#gets the transfer function of the butterworth filter
transferFunction = signal.TransferFunction(b, a)
print("Transfer Function: ", transferFunction)

#gets the normalized frequency response
zp.freq(transferFunction, type='solid', grid=True, name='Order 3 Lowpass Filter')


#gets the Pole-Zero plot
zp.pz(transferFunction)


#sets the signal length
signalLength = 400

#creates a signal for the matrix
x_signal = np.zeros((signalLength, 1))

for i in range(signalLength):
    x_signal[i][0] = np.cos((np.pi/17)*i + np.pi/2)



#creates a new y signal
y_signal_correct = np.zeros((signalLength, 1))

#sets the number of coefficients
numCoefficients = N + M + 1

#sets the parameters for the noise mean and variance
noise_mean = 0.0
noise_variance = 0.01


#creates the a_k temp vector
a_k = np.zeros((1, numCoefficients))

A = np.zeros((signalLength, numCoefficients))

#attempts to perform the summation to create each output sample
for n in range(signalLength):
    #iterates through each of a coefficients

    #resets a_k to zeros
    a_k = np.zeros((1, numCoefficients))
    
    #creates the a sum variable
    a_sum = 0.0
    #iterates through the previous output signal samples
    for k in range(1,N+1):
        #checks to make sure that the y sample is within range
        if n - k >= 0:
            #adds the feedback sums
            a_sum = a_sum + a[k]*y_signal_correct[n-k][0]
            #adds the y signal sample to the a_k vector. These need to
            #be negated, because of the misrepresented part
            a_k[0][k-1] = -1.0*y_signal_correct[n-k][0]


    #creates the b sum variable
    b_sum = 0.0
    #iterates through the input signal
    for k in range(0,M+1):
        #checks to make sure that the x sample is within range
        if n - k >= 0:
            #adds the input signal sums
            b_sum = b_sum + b[k]*x_signal[n-k][0]
            #adds the x_signal sample to the a_k vector
            a_k[0][k+N] = x_signal[n-k][0]
    
    #adds a_k to the leading edge of the total A matrix
    A[n,:] = a_k[0,:]


    #sets the y output signal to the sum, and adds a certain level of noise
    y_signal_correct[n][0] = -a_sum + b_sum + np.random.normal(loc=noise_mean, scale=noise_variance)




#gets the error signal between the y_signal and the y_signal_new




#prints out the error over various positional indecies
print("Y signal new: ", y_signal_correct[40:45])


plt.figure()

#plt.plot(x_signal[0:100])
plt.plot(y_signal_correct[0:1000])



#in order to prove that I actually am doing things right
#I used matlab to filter the exact same x signal and
matlab_y_signal = np.loadtxt("matlab_y_signal.csv", dtype='float')
matlab_y_signal = matlab_y_signal[0:signalLength]
matlab_y_signal_length = np.size(matlab_y_signal)
#print(matlab_y_signal_length)
matlab_y_signal = matlab_y_signal.reshape((matlab_y_signal_length,1))
#print(matlab_y_signal)

plt.plot(matlab_y_signal[0:1000])

plt.legend(["Y signal with Noise", "Y signal without noise"])


#writes A, x, and y to csv
np.savetxt("A.csv", A, delimiter=",")
np.savetxt("X.csv", x_signal, delimiter=",")
np.savetxt("Y.csv", y_signal_correct, delimiter=",")


#calculates x using the batch pseudoinverse
x_star = np.linalg.inv(A.T @ A) @ A.T @ y_signal_correct
#prints the coefficients
print("Coefficients: \n", coefficients)

#prints the pseudoinverse and the error
print("x star: \n", x_star)

#prints the error
error = x_star - coefficients
print("Error: \n", error)




# %%
