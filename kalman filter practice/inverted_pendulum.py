
#%%
#this file looks at the observability and implements a kalman filter for the state of the inverted pendulum on a cart


import numpy as np

import control as ctrl


#mass of the cart
m = 1
#mass of the pendulum
M = 5
#length of the pendulum
L = 2
#gravity constant
g = -9.8

d = 1


s = -1

#gets the A matrix

A = np.array([[0, 1.0, 0, 0],
              [0, -d/M, -m*g/M, 0],
              [0, 0, 0, 1],
              [0, -s*d/(M*L), -s*(m+M)*g/(M*L), 0]])

#gets the B matrix
B = np.array([[0],
              [1/M],
              [0],
              [s*1/(M*L)]])

#gets the C matrix

C = np.array([[1.0, 0.0, 0.0, 0.0]])


#gets the observability matrix with A and C

O = ctrl.obsv(A, C)

#prints the observability matrix
print("Observability matrix")
print(O)

#prints the rank of the observability matrix
print("Observability matrix Rank", np.linalg.matrix_rank(O))
# %%
