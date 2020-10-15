#Levy Flight Foraging Hypothesis Simulation 
#computing survival rates over a large parameter space

import numpy as np
from matplotlib import pyplot as plt

def take_levy_steps(x_0,N,b,m,max_energy):
    
    # x_0 : Initial Position
    # N : Number of Steps
    # b : Levy coefficient
    # m : Maximum step-size
    # max_energy : maximum energy of animal
    
    
    # defining heavy tailed pdf using levy-coefficient and max step-size
    t = np.linspace(0.51,m,10*N)
    s = (1/np.pi)*(1/((b)+(t-1)**2))
    
    #sample from pdf 
    stepsizes = np.random.choice(t,size = N, p = s/s.sum())
    directions = np.random.uniform(low = 0, high = 2*np.pi, size = N)
    
    #polar to cartesian conversion
    steps = polar_to_cart(stepsizes, directions)
    pos_track = x_0
    pos = x_0
    cost = np.zeros(N)
    
    #tracking position
    for i in range(0,N):
        pos = pos + steps[:,i]
        pos_track = np.vstack((pos_track,pos))
        cost[i] = (b)**0.5*(steps[0,i]**2 + steps[1,i]**2)**0.5
        if cost.sum() > max_energy:
            break
        
    return pos_track

def polar_to_cart(r,theta):
    
    #converting polar coordinates to cartesian coordinates
    
    z = r * np.exp( 1j * theta )
    x_coord = np.real(z)
    y_coord = np.imag(z)
    
    return np.array([x_coord, y_coord])


#%%
    
b_s = np.logspace(-5,4,50)      #range of levy coefficients to test
m_s = np.linspace(2,300,50)     #range of max step sizes to test
b_vs_m = np.zeros((50,50))

#initialising simulation
x_0 = np.array([0,0])
N = 500
max_energy = 750
its = 100
    
#finding survival rates of every combination of b vs m
for i in range(0,50):
    for j in range(0,50):
        for n in range(0,its):
            pos_track = take_levy_steps(x_0,N,b_s[i],m_s[j],max_energy)
            
            f = np.random.uniform(low = -100,high = 100, size = 8)
            food = np.array([[f[0],f[1]],[f[2],f[3]],[f[4],f[5]],[f[6],f[7]]])
            r = np.random.uniform(low = 5,high = 20, size = 4)
            pt = pos_track
            for k in range(4):
                if (((pt-food[k,:])[:,0]**2+(pt-food[k,:])[:,1]**2)**0.5).min()<r[k]:
                    b_vs_m[i,j] = b_vs_m[i,j] + 1

b_vs_m = b_vs_m/its

#plotting contour plot of survival rates over different levy coefficients and 
#maximum step sizes
    
b_s = np.logspace(-5,4,50)
m_s = np.linspace(2,300,50)    

fig1, ax2 = plt.subplots(constrained_layout=True)

CS = ax2.contourf(m_s,
                  b_s,
                  b_vs_m,
                  10,
                  cmap='viridis',
                  levels = 7,
                  extent=[2,300,0.0001,10],
                  vmin=0,
                  vmax=0.8)
cbar = fig1.colorbar(CS)
plt.yscale('log')
plt.xscale('log')
plt.ylabel("lévy-coefficient")
plt.xlabel("Maximum Stepsize")
cbar.ax.set_ylabel('Survival Rate')
plt.scatter(2.12,0.001, color = 'k', marker = 'x')

plt.show()