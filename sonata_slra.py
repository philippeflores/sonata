import numpy as np
import quaternion as qt
import scipy
import scipy.sparse
import scipy.sparse.linalg
import bispy as bsp

from sonata_base import *

def signal_to_hankel(signal,L):
    
    array_dimension = np.shape(signal)
    
    if len(array_dimension) == 1:
        signal_as_vec = np.reshape(signal,[-1,1])
        N = len(signal_as_vec)
    else:
        raise ValueError("The signal to be Hankelized must be a vector.")
    
    if L>0:
        K = N-L+1
    else:
        raise ValueError("The size of the Hankel matrix is not valid. 'L' must be greater than 0.")
    
    hankel_matrix = np.zeros([L,K],dtype = np.dtype(signal_as_vec[0,0]))
    for l in range(L):
        for k in range(K):
            hankel_matrix[l,k] = signal_as_vec[l+k,0]
            
    return hankel_matrix


def hankel_to_signal(hankel_matrix):
    
    array_dimension = np.shape(hankel_matrix)
    
    if len(array_dimension) == 2:
        L = array_dimension[0]
        K = array_dimension[1]
        N = L+K-1
    else:
        raise ValueError("The input Hankel matrix must be a matrix.")
    
    signal = np.zeros(N, dtype=np.dtype(hankel_matrix[0,0]))
    for n in range(K-1,-L,-1):
        signal[np.mod(N-(n+L),N)] = np.mean(np.diagonal(np.fliplr(hankel_matrix),n))
        
    return signal


def sonata(y,R, L=-1, M_0=[], number_outer_iterations = 200, number_inner_iterations = 5, tolerance_outer = 1e-6, tolerance_inner = 1e-4):
    N = np.shape(y)[0]
    
    if L<0:
        L = int(np.floor(N/2))
        
    if np.shape(M_0)[0]==0:
        H = signal_to_hankel(y,L)

        U,D,Vh = SVDH(H,R = R)

        M_0 = np.zeros([N,R],dtype = complex)

        for r in range(R):
            y_r = hankel_to_signal(D[r]*ldot(np.reshape(U[:,r],[-1,1]),np.reshape(Vh[r,:],[1,-1])))
            a, theta, chi, phi = bsp.utils.quat2euler(y_r)
            
            phi_0 = phi[0]
            a_0 = a[0]
            
            M_0[:,r] = a*np.exp((phi-phi_0)*1j)/a_0

        
    M_hat = np.array(M_0)
    q_hat = np.zeros(R,dtype = qt.quaternion)

    index_outer_iterations = 0
    flag_outer = 0
    while flag_outer == 0:
        
        q_old = np.array(q_hat)
        
        q_hat = rdot(np.linalg.pinv(M_hat),y)
        
        M_old = np.array(M_hat)
        
        for r in range(R):
            
            y_tilde = np.array(y)
            for s in range(R):
                if s!=r:
                    y_tilde = y_tilde-rdot(M_hat[:,s],q_hat[s])
            z_r_hat = quaternion_to_complex(q_hat[r].inverse()*y_tilde)[0]
            
            flag_inner = 0
            index_inner_iterations = 0
            while flag_inner==0:
                z_r_old = np.array(z_r_hat)
                
                H = signal_to_hankel(z_r_hat,L)
                U,D,V = scipy.sparse.linalg.svds(H,1)
                H_hat = (np.reshape(U[:,0],[-1,1])*D[0])@np.transpose(np.reshape(V[0,:],[-1,1]))
                z_r_hat = hankel_to_signal(H_hat)
                
                if np.linalg.norm(z_r_hat-z_r_old)**2<tolerance_inner:
                    flag_inner = index_inner_iterations
                
                if index_inner_iterations<number_inner_iterations:
                    index_inner_iterations = index_inner_iterations+1
                else:
                    flag_inner = number_inner_iterations
            q_hat[r] = ldot(q_hat[r],z_r_hat[0])
            z_r_hat = z_r_hat/z_r_hat[0]
            M_hat[:,r] = z_r_hat
            
        if np.linalg.norm(M_old-M_hat)**2<tolerance_outer and np.linalg.norm(abs(q_old-q_hat))**2<tolerance_outer:
            flag_outer = index_outer_iterations
        
        if index_outer_iterations<number_outer_iterations:
            index_outer_iterations = index_outer_iterations+1
        else:
            flag_outer = number_outer_iterations
        
    return M_hat, q_hat, flag_outer
    

def quaternion_matrix_rank(X):
    chiX = quaternion_to_adjoint(X)
    rk = np.linalg.matrix_rank(chiX)
    return int(rk/2)

def SVDH(A, R = -1,splitting = "symp"):
    '''
    Function that computes the SVD of a quaternion 
    matrix A = A_1 + A_2 j
    
    input: A_1 and A_2, complex matrices of size NxM
    
    output: 
    U1,U2: two complex matrices of left singular vectors forming the quaternion matrix 
    U = U1 + U2 j
    Vh1,Vh2: two complex matrices of right singular vectors forming the quaternion matrix 
    Vh = Vh1 + Vh2 j
    D: real singular values (dimension = min(N,M))
    '''
    if R<0:
        R = quaternion_matrix_rank(A)
    
    N, M = np.shape(A)
    
    #forming the complex representation
    ChiA = quaternion_to_adjoint(A,splitting = splitting)
    #computing the SVD of ChiA
    Uchi,Dchi,Vhchi=np.linalg.svd(ChiA)
    
    Vchi=np.conj(Vhchi.T) # to be used for extraction of singular vectors
    
    # allocating singular values of ChiA to those of A
    D=Dchi[::2]
    # allocating singular vectors of ChiA to those of A
    UCH = Uchi[:,::2]  
    VC = Vchi[:,::2]
    
    if splitting.lower()=="cd":
        U1 = UCH[0:N,:]
        U2 = -np.conjugate(UCH[N:2*N,:])
        V1 = VC[0:M,:]
        V2 = -np.conjugate(VC[M:,:])
    else:
        U1 = UCH[0:N,:]
        U2 = UCH[N:,:]
        V1 = VC[0:M,:]
        V2 = VC[M:,:]
    
    # transpose conjugate back so that A = U@D@Vh in H
    Vh1=np.conj(V1.T)
    Vh2=-V2.T
    
    D = D[:R]
    U = complex_to_quaternion(U1[:,:R],U2[:,:R],splitting = splitting)
    Vh = complex_to_quaternion(Vh1[:R,:],Vh2[:R,:],splitting = splitting)
    
    return U,D,Vh
