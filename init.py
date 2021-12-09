import numpy as np
import constants as c

def Circular_motion(D, R, m):
    N = np.shape(D)[1]
    M = N*m/2
    V = (c.G*M/R)**0.5
    for i in range(N):
        D[0:3,i] = np.array([R*np.cos(2*np.pi*i/N), R*np.sin(2*np.pi*i/N), 0])
        D[3:6,i] = np.array([-V*np.sin(2*np.pi*i/N), V*np.cos(2*np.pi*i/N), 0])
    D[6,:] = m
    return D

def Sun_Earth(D): 
    # Sun
    D[:3,0] = 0
    D[3:6,0] = 0
    D[6,0] = 2e30
    # Earth
    D[:3,1] = np.array([1.5e11, 0, 0])
    D[3:6,1] = np.array([0, 2.4e4, 0])
    D[6,1] = 6e24
    return D
