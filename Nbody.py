import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from constants import *
import init

# Let's use MKS
N = int(10) # Number of Shells
T = 86400*365*500

D = np.zeros((7, N))


D = init.Circular_motion(D, 1.5e11, 6e24)

def Acc(D):
    a = np.zeros((3, N))
    for i in range(N):
        M = D[6,:]
        r_ij = D[0:3,:] - np.outer(D[0:3,i], np.ones(N))
        r_ij[:,i] = 1 # To avoid divided by 0
        a_ij = G*M/np.linalg.norm(r_ij, axis=0)**3*r_ij
        a_ij[:,i] = 0
        a_i = np.sum(a_ij, axis=1)
        a[:,i] = a_i
    return a

def function(t, y):
    y = y[:,-1].reshape(7,N)
    r = y[0:3,:]
    v = y[3:6,:]
    M = D[6,:]
    dfdt = np.row_stack((np.concatenate((v, Acc(y)), axis=0),np.zeros(N))).flatten()
    return dfdt

sol = solve_ivp(fun = function,
                t_span = (0, T),
                y0 = D.flatten(),
                method = 'LSODA',
                vectorized=True
                )

print(np.shape(sol.y))
SOL = np.ones((7, N, np.shape(sol.y)[1]))
for i in range(np.shape(sol.y)[1]):
    SOL[:,:,i] = sol.y[:,i].reshape((7,N))
for i in range(N):
    plt.plot(SOL[0,i,:], SOL[1,i,:])
#plt.ylim([0,1])
#plt.xlabel("Time")
#plt.ylabel("R")
#plt.savefig("./Result.png", dpi=300)
plt.show()
