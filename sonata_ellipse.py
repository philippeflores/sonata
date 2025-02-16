import numpy as np
import quaternion as qt
import bispy as bsp
from sonata_base import *

def generate_random_damped_ellipse(N,amplitude_min = 0.5, amplitude_max = 2, frequency_min = 8, frequency_max = 64, damping_min = 1, damping_max = 10):

    if damping_max<damping_min:
        raise ValueError("The parameter 'damping_min' is greater than 'damping_max'.")
    if amplitude_max<amplitude_min:
        raise ValueError("The parameter 'amplitude_min' is greater than 'amplitude_max'.")
    if frequency_max<frequency_min:
        raise ValueError("The parameter 'frequency_min' is greater than 'frequency_max'.")
    
    a_0 = amplitude_min+(amplitude_max-amplitude_min)*np.random.rand()
    chi = -(np.pi/4)+np.random.rand()*np.pi/2
    theta = np.random.rand()*np.pi
    phi0 = -np.pi+np.random.rand()*2*np.pi
    
    q = bsp.utils.euler2quat(a_0,theta,chi,phi0)

    f = (frequency_min+(frequency_max-frequency_min)*np.random.rand())/N
    d = (damping_min+(damping_max-damping_min)*np.random.rand())/N
    
    mu = np.exp(-d)*np.exp(2*np.pi*f*1j)
    M = mu**np.arange(N)
        
    y = rdot(M,q)
    
    return y, M, q
     

def generate_ellipse_mixture(N,R,amplitude_min = 0.5, amplitude_max = 2, frequency_min = 8, frequency_max = 64, damping_min = 1, damping_max = 10):

    if damping_max<damping_min:
        raise ValueError("The parameter 'damping_min' is greater than 'damping_max'.")
    if amplitude_max<amplitude_min:
        raise ValueError("The parameter 'amplitude_min' is greater than 'amplitude_max'.")
    if frequency_max<frequency_min:
        raise ValueError("The parameter 'frequency_min' is greater than 'frequency_max'.")

    M = np.zeros([N,R], dtype = complex)
    q = np.zeros(R, dtype = qt.quaternion)
    for r in range(R):
        y, M[:,r], q[r] = generate_random_damped_ellipse(N,amplitude_min = amplitude_min, amplitude_max = amplitude_max, frequency_min = frequency_min, frequency_max = frequency_max, damping_min = damping_min, damping_max = damping_max)
    
    y = rdot(M,q)
    
    return y, M, q


def add_quaternion_white_noise(y,snr):
    N = np.shape(y)[0]
    
    b_complex = bsp.signals.bivariatewhiteNoise(N,np.mean(abs(y)**2)/(2*snr))
    bh = bsp.timefrequency.Hembedding(b_complex)
    b = bh.Hembedding
    
    y_noised = y+b
    
    return y_noised, b 