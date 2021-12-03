import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Let's use MKS
N = int(1e1) # Number of Shells
rho = 0.01 # Density of the medium
R = 1
T = 10
G = 6.67e-11

D = np.zeros((7, N))
D[0:3,:] = np.random.random((3,N)) # Linear separation shells
D[3:6,:] = np.random.random((3,N)) # Linear separation shells
D[6,:] = 1 # Mass of each particle

def Acc(D):
    print(D[0:3,:])
    a = np.zeros((3, N))
    for i in range(N):
        r = D[0:3,:] - D[0:3,i]*np.ones((3, N))
        a[i,:] = np.sum(-G*D[7,:]*D[7,i]/np.linalg.norm(r)**3*r, axis=1)
    return a

def function(t, y):
    y = y[:,-1].reshape(7,N)
    r = y[0:3,:]
    v = y[3:6,:]
    #D[0:3,:] = r
    #D[3:6,:] = v
    dfdt = np.concatenate(v, Acc(y), axis=0)
    #dxdt = v
    #dvdt = Acc(D)
    return dfdt

sol = solve_ivp(fun = function,
                t_span = (0, T),
                y0 = D.flatten(),
                method = 'LSODA',
                vectorized=True
                )

print(np.shape(sol.y))
#x = sol.y[0,:]
#y = sol.y[1,:]
#z = sol.y[2,:]
#V = sol.y[N:]
#t = sol.t

#for i in range(N):
#    plt.plot(t, np.abs(R[i,:]))
#plt.ylim([0,1])
#plt.xlabel("Time")
#plt.ylabel("R")
#plt.savefig("./Result.png", dpi=300)
#plt.show()
