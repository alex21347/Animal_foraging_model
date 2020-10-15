#Evolutionary Algorithm for Levy Flight Foraging Hypothesis

import numpy as np
from matplotlib import pyplot as plt

def polar_to_cart(r,theta):
    
    #Converts polar coordinates to cartesian coordinates
    
    z = r * np.exp( 1j * theta )
    x_coord = np.real(z)
    y_coord = np.imag(z)
    return np.array([x_coord, y_coord])

def take_levy_steps(x_0,N,b,m,max_energy):
    
    #take a sequence of levy steps
    
    #sample from heavy tailed distribution
    t = np.linspace(0.51,m,10*N)
    s = (1/np.pi)*(1/((b)+(t-1)**2))
    stepsizes = np.random.choice(t,size = N, p = s/s.sum())
    directions = np.random.uniform(low = 0, high = 2*np.pi, size = N)
    steps = polar_to_cart(stepsizes, directions)
    
    #tracking position
    pos_track = x_0
    pos = x_0
    cost = np.zeros(N)
    for i in range(0,N):
        pos = pos + steps[:,i]
        pos_track = np.vstack((pos_track,pos))
        cost[i] = (b)**0.5*(steps[0,i]**2 + steps[1,i]**2)**0.5
        if cost.sum() > max_energy:
            break
        
    return pos_track
    
b = 0.001                        #starting levy coefficient
m = 2                            #starting maximum step-size
d1 = 10                          #gradient descent factor in b-axis   
d2 = 5                           #gradient descent factor in m-axis
maxit = 3                        #number of iterations/'generations'
N = 500                          #max number of steps
max_energy = 750                 #max energy
x_0 = np.array([0,0])            #origin of walk
its = 250                        #number of simulations for each proposed mutation
trace = np.zeros((maxit,2))      #this was used to create the travel path of the evo alg.

for n in range(maxit):
    
    #construct 25 nearby mutations in parameter space
    #note : diversity of mutations depend directly on d1 & d2, see below
    
    trace[n,:] = [m,b]
    success_rates = np.zeros((5,5))
    b_s = np.array([(1/d1)*b,0.5*(((1/d1)*b)+b),b,0.5*((d1*b)+b),b*d1])
    m_s = np.array([(1/d2)*m,0.5*(((1/d2)*m)+m),m,0.5*((d2*m)+m),m*d2])
    
    #simulate each mutation over many iterations to approximate survival rate
    for i in range(5):
        for j in range(5):
            for n in range(0,its):
                #find travel path
                pos_track = take_levy_steps(x_0,N,b_s[i],m_s[j],max_energy)
                
                #constructing randomised food patches
                f = np.random.uniform(low = -100,high = 100, size = 8)
                food = np.array([[f[0],f[1]],[f[2],f[3]],[f[4],f[5]],[f[6],f[7]]])
                r = np.random.uniform(low = 5,high = 20, size = 4)
                
                #if the animal hits the food, mark as success
                pt = pos_track
                for k in range(4):
                    if (((pt-food[k,:])[:,0]**2+(pt-food[k,:])[:,1]**2)**0.5).min()<r[k]:
                        success_rates[i,j] = success_rates[i,j] + 1
            success_rates[i,j] = success_rates[i,j]/its  
    print(b,m)
    print()
    print(success_rates)
    print()
    
    #the most optimal mutation 'passes on its genes'
    b = b_s[np.where(success_rates == success_rates.max())[0][0]]               
    m = m_s[np.where(success_rates == success_rates.max())[1][0]]    
    print('------------')
                    
    
    #if all of the new mutations were less optimal than the previous generation,
    #the below code reduces the diversity of mutations to be closer to the previous 
    #generation
    
    if np.where(success_rates == success_rates.max())[0][0] == 2:
        d1 = 1+0.5*np.abs(d1-1)
        
    if np.where(success_rates == success_rates.max())[1][0] == 2:
        d2 = 1+0.5*np.abs(d2-1)

#plotting the travel path of the evolved animal
for u in range(50):
        
    #plotting travel path
    pos_track = take_levy_steps(np.array([0,0]),500,b,m,750)
    f = np.random.uniform(low = -100,high = 100, size = 8)
    food = np.array([[f[0],f[1]],[f[2],f[3]],[f[4],f[5]],[f[6],f[7]]])
    r = np.random.uniform(low = 5,high = 20, size = 4)
            
    fig, ax = plt.subplots(figsize = (6,6))
    
    #plotting food
    for i in range(4):
        circle2 = plt.Circle((food[i,0],food[i,1]), r[i], color='green', alpha = 0.5)
        ax.add_artist(circle2)
        
    ax.set_xlim((-100, 100))
    ax.set_ylim((-100, 100))
    ax.plot(pos_track[:,0],
            pos_track[:,1],
            color = 'k', 
            linewidth = 1.1,
            linestyle = 'dashed')
    #ax.axis('equal')
    ax.grid(color='k', linestyle='--', linewidth=0.5,alpha = 0.4)
    plt.show()