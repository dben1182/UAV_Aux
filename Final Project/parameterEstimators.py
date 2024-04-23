#parameter Estimation code. This code implements an algorithm to obtain an estimate 
#of the Aerodynamic parameters of the aerosonde while in flight.
#we will need to implement a launch file that causes the system to undergo
#controls that excite the various modes needed for proper system characterization.
import numpy as np




#creates a class for a stationary Kalman Filter to implement the various modes here.


#there will be an independent stationary kalman filter class instantiated for every
#input y into our system. Each single variable sensor input to our system will have

class stationaryKalmanFilter:


    #creates the initialization function,
    #takes as an argument
    #ts: the time sample
    #x_init: the initial x_estimate
    #P_init: the initial P estimate 
    #y_length: the length of the measurements per sample to be input to the system.
    def __init__(self, ts, x_init, P_init, y_length):
        #sets the sample rate of the observer
        self.Ts = ts
        #sets the x_star to the initialized parameter
        self.x_star = x_init
        #gets the x_length, or the number of x coefficients to be estimated
        self.x_length = np.size(x_init)
        #sets the P initialized
        self.P = np.copy(P_init)
        #sets the length of the y matrix
        self.y_length = y_length

        #creates the y vector, which contains the full signal of y
        self.y = np.zeros((1, 1))

        #creates the initial A_N matrix
        self.A_N_initialization = np.zeros((y_length, self.x_length))

        #creates the flag which sets initializing to True.
        #once we are done initializing with batch least squares,
        #we don't add to the A_N matrix
        self.initializing = True


        #creates the counter
        self.counter = 0

    #creates the update function, which is split into two portions,
    #one for initialization, which uses batch least squares, until
    #we achieve a full rank matrix, and a recursive least squares algorithm.
    #it calls the batch initialization function to initialize the matrix

    #arguments:
    #1. y_n is the y_n vector of the current time's sensor samples
    #2. a_n is the matrix of the data, which characterizes the system for the given input

    #returns

    def update(self, y_n, a_n):
        
        #checks if we have a full rank P matrix based on the current A_N matrix
        if self.initializing:
            #checks if we need to create or append to the initialization
            if self.counter == 0:
                #sets A_N to the first input transposed
                self.A_N_initialization = a_n.T
                #sets the y to the first signals
                self.y = y_n

            #otherwise we append the current input a_n onto the A_N initialization function
            else:
                #appends the new a_n to the A_N
                self.A_N_initialization = np.append(self.A_N_initialization, a_n.T, axis=0)
                #appends the new y_n
                self.y = np.append(self.y, y_n, axis=0)

            #gets the P_N_inverse
            P_N_inv = self.A_N_initialization.T @ self.A_N_initialization

            #gets the rank of P_N to see if it is invertible
            P_N_inv_rank = np.linalg.matrix_rank(P_N_inv)
            
            #checks if the matrix rank is equal to the length of x, such that P_N_inv is invertible
            if P_N_inv_rank == self.x_length:
                #if this is true, we set the P matrix, and the initial x_star for the kalman filter
                self.P = np.linalg.inv(P_N_inv)
                #and gets the initial x star
                #print("P: ", self.P)
                #print("y: ", self.y)

                self.x_star = self.P @ self.A_N_initialization.T @ self.y
                #now that we have initialized, we set the initializing flag to false, so we run the
                #recursive least squares next time.
                self.initializing = False
                print("x star intitial: ", self.x_star )
            
            #increments the counter
            self.counter += 1

        #otherwise if we are not initializing, we perform the normal recursive least squares,
        #or stationary Kalman Filter approach
        else:   
            #with the inputs and the current state, gets the kalman gain, k_N
            K_N = (self.P @ a_n) @ np.linalg.inv(np.eye(self.y_length) + np.transpose(a_n) @ self.P @ a_n)

            #gets the new P matrix
            self.P = self.P - K_N @ a_n.T @ self.P

            #gets the update for x_star
            self.x_star = self.x_star + K_N @ (y_n - np.transpose(a_n) @ self.x_star)
