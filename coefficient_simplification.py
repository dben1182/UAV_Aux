#%%
#this file implements using sympy to drastically reduce my linear algebra workload for this project

#imports the needed libraries
import sympy as sp
import numpy as np

from IPython.display import display, Latex

#creates the rho, which is air density
rho = sp.symbols('rho')

#creates Va, the airspeed
Va = sp.symbols('Va')

#creates S, the surface area of the wing
S = sp.symbols('S')

#creates m, the mass of the mav
m = sp.symbols('m')

#sets c
c = sp.symbols('c')

#sets q
q = sp.symbols('q')

#sets delta_e, for the elevator command
delta_e = sp.symbols('delta_e')

#defines the constants that are out front of the equation
constants = (rho*(Va**2)*S)/(2*m)

#defines alpha, or the angle of attack
alpha = sp.symbols('alpha')

print("Constants: ")
display(constants)

#defines the mixing matrix, which mixes all the coefficients together
mixingMatrix = sp.Matrix([[1, (c*q)/(2*Va), delta_e]])

print("Mixing matrix: ")
display(mixingMatrix)

#gets the trig matrix
trigMatrix = sp.Matrix([[-sp.cos(alpha), sp.sin(alpha), 0, 0, 0, 0],
                        [0, 0, -sp.cos(alpha), sp.sin(alpha), 0, 0],
                        [0, 0, 0, 0, -sp.cos(alpha), sp.sin(alpha)]])

print("Trig matrix")
display(trigMatrix)

#gets the alpha matrix
alphaMatrix = sp.Matrix([[1, alpha, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, alpha, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1]])

print("Alpha Matrix")
display(alphaMatrix)

#creates the C_D0 symbol
C_D0 = sp.symbols('C_{D0}')
display(C_D0)
#creates the C_D alpha symbol
C_D_alpha = sp.symbols('C_{D\\alpha}')
display(C_D_alpha)

C_L0 = sp.symbols('C_{L0}')

C_L_alpha = sp.symbols('C_{L\\alpha}')

C_Dq = sp.symbols('C_{Dq}')

C_Lq = sp.symbols('C_{Lq}')

C_D_delta_e = sp.symbols('C_{D_{\\delta_e}}')

C_L_delta_e = sp.symbols('C_{L_{\\delta_e}}')

#creates the coefficient array
C_coefficients = sp.Matrix([[C_D0],
                            [C_D_alpha],
                            [C_L0],
                            [C_L_alpha],
                            [C_Dq],
                            [C_Lq],
                            [C_D_delta_e],
                            [C_L_delta_e]])

display(C_coefficients)

#creates the whole row vector that multiplies with the coefficients matrix
wholeMixingMatrix = mixingMatrix*trigMatrix*alphaMatrix

display(wholeMixingMatrix)


# %%

