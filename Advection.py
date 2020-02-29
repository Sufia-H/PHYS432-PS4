# -*- coding: utf-8 -*-
"""
Solving the advection equation
using both Forward-Time Central-Space
and Lax-Friedrichs methods

@author: Sufia Hashim
Feb 28th 2020
"""
## Importing relevant modules
import numpy as np
import matplotlib.pyplot as plt

## setting up the grid and initialising variables
Ncells = 100 #number of cells in grid
steps = 5000 #number of time steps
dt = 3 #time step
dx = 1 #position step
x = np.arange(Ncells)*dx
u = -0.1 #velocity 
factor = u*dt/(2*dx) 

## Defining initial conditions
f1 = np.copy(x)*1.0/Ncells #FTCS
f2 = np.copy(x)*1.0/Ncells #Lax-Friedrichs (L-F)

## Setting up the plot
plt.ion()
fig, axes = plt.subplots(1,2)
axes[0].set_title('FTCS')
axes[1].set_title('Lax-Friedrichs')
for ax in axes:
    ax.set_xlim([0, Ncells])
    ax.set_ylim([0,2])

## Plotting initial state in background for reference
axes[0].plot(x,f1, 'b-', label='Initial state') 
axes[1].plot(x,f2, 'b-', label='Initial state') 

## Updating the plots
plt1,  = axes[0].plot(x, f1, 'r.', label='Time-evolved state') 
plt2,  = axes[1].plot(x, f2, 'r.', label='Time-evolved state') 
plt.legend()
fig.canvas.draw()

## While-loop to calculate the value of f(x,t)
# using both FTCS and L-F methods
# and updating the plot for each iteration
count = 0 
while count < Ncells:
    #keeping first and last cells constant in time
    f1[1:Ncells-1] = f1[1:Ncells-1] - factor*(f1[2:] - f1[:Ncells-2]) #FTCS
    f2[1:Ncells-1] = 0.5*(f2[2:] + f2[:Ncells-2]) - factor*(f2[2:] - f2[:Ncells-2]) #L-F
   
    ## Updating the plots
    plt1.set_ydata(f1)
    plt2.set_ydata(f2)
    fig.canvas.draw()
    plt.pause(0.001)
    count +=1 

