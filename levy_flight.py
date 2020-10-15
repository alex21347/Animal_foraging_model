#Simulating the Levy flight 

import numpy as np
from matplotlib import pyplot as plt

x = np.array([0,0]) #origin
N = 2000            #number of steps

def polar_to_cart(r,theta):
    
    #Converts polar coordinates to cartesian coordinates
    
    z = r * np.exp( 1j * theta )
    x_coord = np.real(z)
    y_coord = np.imag(z)
    return np.array([x_coord, y_coord])

def take_levy_steps(x_0,N,L):
    
    #takes a sequence of N levy steps from one of 3 Levy stable pdfs
    
    t = np.linspace(0.51,50,10*N)
    
    if L == 1:    #Lévy-Smirnov distribution
        s = (1/np.sqrt(2*np.pi))*(t-0.5)**(-3/2)*(np.exp(-1/(2*(t-0.5))))
        
    if L == 2:    # Cauchy distribution
        s = (1/np.pi)*(1/(0.01+(t-1)**2))
        
    if L == 3:    # Gaussian distribution
        s = (1/np.sqrt(2*np.pi))*(np.exp(((t-1)**2)/-2))
         
        
    #sampling distribution
    stepsizes = np.random.choice(t,size = N, p = s/s.sum())
    directions = np.random.uniform(low = 0, high = 2*np.pi, size = N)
    steps = polar_to_cart(stepsizes, directions)
    pos_track = x_0
    pos = x_0
    
    #tracking position
    for i in range(0,N):
        pos = pos + steps[:,i]
        pos_track = np.vstack((pos_track,pos))
    return pos_track


#plotting travel path
pos_track = take_levy_steps(x,N,2)
fig = plt.figure(figsize = (6,6))
plt.plot(pos_track[:,0],pos_track[:,1], color = 'k', linewidth = 1)
plt.axis('equal')
plt.scatter([-1,2],[-0.7,4],color ='white')
plt.grid(color='k', linestyle='--', linewidth=0.5,alpha = 0.4)
plt.show()