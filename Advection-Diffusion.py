"""
Solving the advection-diffusion equation
with advection term solved using Lax-Friedrichs method
and diffusion term using implicit Operator Splitting method

@author: Sufia Hashim
Feb 28th 2020
"""

## Importing relevant modules
import numpy as np
import matplotlib.pyplot as plt

## Setting up the grid and initialising variables
Ncells = 100 #number of cells in grid
steps = 5000 #number of time steps
dt = 10 #time step 
dx = 1 #position step
u = -0.1 #velocity 
D = np.array([1, 5]) #2 diffusion co-efficients
factor = u*dt/(2*dx) #for L-F
beta = D*dt/dx**2 #for implicit method
x = np.arange(Ncells)*dx

## Defining initial conditions
# total f(x,t) with both advection and diffusion terms considered
ftotal = np.array([np.copy(x)*1.0/Ncells, np.copy(x)*1.0/Ncells]) 

## Setting up the plot
plt.ion()
fig, axes = plt.subplots(1,2)
axes[0].set_title('Advection-Diffusion with D = {}'.format(D[0]))
axes[1].set_title('Advection-Diffusion with D = {}'.format(D[1]))
for ax in axes:
    ax.set_xlim([0, Ncells])
    ax.set_ylim([0,2])

## Plotting initial state in background for reference
axes[0].plot(x,ftotal[0], 'b-')
axes[1].plot(x,ftotal[1], 'b-')    

## Updating the plots
plt1,= axes[0].plot(x, ftotal[0], 'r.') 
plt2,= axes[1].plot(x, ftotal[1], 'r.') 
plt.legend()
fig.canvas.draw()

## Initialising the tri-diagonal diffusion matrix A with elements -beta, (2*beta + 1), -beta
A1 = np.eye(Ncells)*(1.0 + 2.0*beta[0]) + np.eye(Ncells, k=1)*-beta[0] + np.eye(Ncells, k= -1)*-beta[0]
A2 = np.eye(Ncells)*(1.0 + 2.0*beta[1]) + np.eye(Ncells, k=1)*-beta[1] + np.eye(Ncells, k= -1)*-beta[1]

## Imposing (fixed) no-slip boundary conditions on both sides of the grid
# for A matrix corresponding to first diffusion coefficient
A1[0][0] = 1 #1st element fixed
A1[0][1] = 0 #ensures no diffusive flux through 1st element
A1[-1][-1] = 1 #last element fixed
A1[-1][-2] = 0
# same for second diffusion coefficient
A2[0][0] = 1
A2[0][1] = 0
A2[-1][-1] = 1
A2[-1][-2] = 0

## While-loop to calculate the value of f(x,t)
# for the advection term using L-F 
# and for the diffusion term using the implicit method
# and updating the plot for each iteration
count = 0
while count < Ncells:
  
    #calculating f(x,t) for diffusion for both coefficients
    ftotal[0] = np.linalg.solve(A1, ftotal[0]) 
    ftotal[1] = np.linalg.solve(A2, ftotal[1])
    
    #updating f(x,t) for advection for both coefficients
    ftotal[0][1:Ncells-1] = 0.5*(ftotal[0][2:] + ftotal[0][:Ncells-2]) - factor*(ftotal[0][2:] - ftotal[0][:Ncells-2])
    ftotal[1][1:Ncells-1] = 0.5*(ftotal[1][2:] + ftotal[1][:Ncells-2]) - factor*(ftotal[1][2:] - ftotal[1][:Ncells-2])

    ## Updating the plots
    plt1.set_ydata(ftotal[0])
    plt2.set_ydata(ftotal[1])
    
    fig.canvas.draw()
    plt.pause(0.001)
    count +=1 