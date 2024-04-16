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

y_accel_x = (rho*(Va**2)*S)/(2*m)

display(y_accel_x)



# %%



# %%
