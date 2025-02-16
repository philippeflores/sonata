import numpy as np
import quaternion as qt
import bispy as bsp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sonata_base import *

def plot_theoretical_signal(t,y, color_th = "#1f77b4"):
    
    N = np.shape(t)[0]
    
    y_1, y_2 = quaternion_to_complex(y)

    plt.figure(figsize = [13,4])
    
    gs = gridspec.GridSpec(2, 6)
    gs.update(hspace=0.5, wspace=1, bottom=0.1, left=0.1, top=0.95, right=0.95)

    ax_u = plt.subplot(gs[0, 2:])
    plt.plot(t,y_1.real, color = color_th)

    ax_v = plt.subplot(gs[1, 2:])
    plt.plot(t,y_2.real, color = color_th)

    ax_uv = plt.subplot(gs[:, :2])
    plt.plot(y_1.real,y_2.real, color = color_th)

    val_lim = np.max(np.abs(ax_u.get_ylim() + ax_v.get_ylim()))
    
    ax_u.set_xlim([t[0], t[N-1]])
    ax_u.set_ylim([-val_lim, val_lim])
    ax_u.set_ylabel("$u(t)$")
    ax_u.legend(["Theory"],loc = 'upper right')
    ax_u.grid()

    ax_v.set_xlim([t[0], t[N-1]])
    ax_v.set_ylim([-val_lim, val_lim])
    ax_v.set_xlabel("$t$")
    ax_v.set_ylabel("$v(t)$")
    ax_v.legend(["Theory"],loc = 'upper right')
    ax_v.grid()

    ax_uv.set_xlim([-val_lim, val_lim])
    ax_uv.set_ylim([-val_lim, val_lim])
    ax_uv.set_ylabel("$v(t)$")
    ax_uv.set_xlabel("$u(t)$")
    ax_uv.legend(["Theory"],loc = 'upper right')
    ax_uv.grid()
    
    plt.show()
    

def plot_noisy_signal(t,y,b, color_th = "#1f77b4", alpha_noisy = 0.2, method_limits = 'noise'):
    
    N = np.shape(t)[0]
    
    y_1, y_2 = quaternion_to_complex(y)
    
    y_noised = y+b
    y_1_noised, y_2_noised = quaternion_to_complex(y_noised)

    plt.figure(figsize = [13,4])
    gs = gridspec.GridSpec(2, 6)
    gs.update(hspace=0.5, wspace=1, bottom=0.1, left=0.1, top=0.95, right=0.95)

    ax_u = plt.subplot(gs[0, 2:])
    plt.plot(t,y_1_noised.real,color = 'black', alpha = alpha_noisy)
    plt.plot(t,y_1.real,color = color_th)

    ax_v = plt.subplot(gs[1, 2:])
    plt.plot(t,y_2_noised.real,color = 'black',alpha = alpha_noisy)
    plt.plot(t,y_2.real,color = color_th)

    ax_uv = plt.subplot(gs[:, :2])
    plt.plot(y_1_noised.real,y_2_noised.real, color ='black', alpha = alpha_noisy)
    plt.plot(y_1.real,y_2.real,color = color_th)

    if method_limits=="noise":
        val_lim = np.max(np.abs(ax_u.get_ylim()+ax_v.get_ylim()))
    else:
        val_lim = 1.1*(np.max([np.reshape(np.abs(y_1.real),[-1,1]),np.reshape(np.abs(y_2.real),[-1,1])]))

    ax_u.set_xlim([t[0], t[N-1]])
    ax_u.set_ylim([-val_lim,val_lim])
    ax_u.set_ylabel("$u(t)$")
    ax_u.legend(["Noised","Theory"],loc = "upper right")
    ax_u.grid()

    ax_v.set_xlim([t[0], t[N-1]])
    ax_v.set_ylim([-val_lim,val_lim])
    ax_v.set_xlabel("$t$")
    ax_v.set_ylabel("$v(t)$")
    ax_v.legend(["Noised","Theory"],loc = "upper right")
    ax_v.grid()

    ax_uv.set_xlim([-val_lim,val_lim])
    ax_uv.set_ylim([-val_lim,val_lim])
    ax_uv.set_xlabel("$u(t)$")
    ax_uv.set_ylabel("$v(t)$")
    ax_uv.legend(["Noised","Theory"],loc = "upper right")
    ax_uv.grid()
    
    plt.show()


def plot_estimated_ellipses(t,M,M_hat,q,q_hat,y_noised, alpha_noisy = 0.2, color_th = "#1f77b4", color_estimated = "#ff7f0e",method_limits = "noise"):
    
    R = np.shape(q)[0]
    N = np.shape(M)[0]
    
    y = rdot(M,q)
    y_1, y_2 = quaternion_to_complex(y)
    
    b = y_noised-y
    b_1, b_2 = quaternion_to_complex(b)

    pseudo_corr_mat = np.zeros([R,R])
    corr_mat = np.zeros([R,R])

    for r in range(R):
        z_1_r, z_2_r = quaternion_to_complex(rdot(M[:,r],q[r]), splitting='cd')
        for s in range(R):
            z_1_r_hat = quaternion_to_complex(rdot(M_hat[:,s],q_hat[s]),splitting='cd')[0]
            pseudo_corr_mat[r,s] = abs(np.sum(np.reshape(z_1_r,[1,-1])@np.reshape(z_1_r_hat,[-1,1])))/(np.linalg.norm(z_1_r)*np.linalg.norm(z_2_r))
            corr_mat[r,s] = abs(np.sum(np.reshape(z_1_r,[1,-1])@np.reshape(np.conjugate(z_1_r_hat),[-1,1])))/(np.linalg.norm(z_1_r)*np.linalg.norm(z_2_r))

    permutation_matrix = pseudo_corr_mat+corr_mat
    permutation = np.zeros(R, dtype = int)
    for r in range(R):
        pos = np.argmax(permutation_matrix)
        row = int(np.floor(pos/R))
        column = np.mod(pos,R)
        permutation[row] = column
        permutation_matrix[:,column] = -1
        permutation_matrix[row,:] = -1

    M_hat_copy = np.array(M_hat)
    q_hat_copy = np.array(q_hat)
    for r in range(R):
        M_hat[:,r] = M_hat_copy[:,permutation[r]]
        q_hat[r] = q_hat_copy[permutation[r]]
        
    y_hat = rdot(M_hat,q_hat)
    y_1_hat, y_2_hat = quaternion_to_complex(y_hat)
    
    plt.figure(figsize = [13,4+4*R])
    gs = gridspec.GridSpec(2+2*R, 6)
    gs.update(hspace=0.5, wspace=1, bottom=0.1, left=0.1, top=0.95, right=0.95)

    ax_u = plt.subplot(gs[0, 2:])
    plt.plot(t,y_1.real+b_1.real,color = 'black', alpha = alpha_noisy)
    plt.plot(t,y_1.real, color = color_th)
    plt.plot(t,y_1_hat.real,':', color = color_estimated)
    
    ax_v = plt.subplot(gs[1, 2:])
    plt.plot(t,y_2.real+b_2.real,color = 'black', alpha = alpha_noisy)
    plt.plot(t,y_2.real, color = color_th)
    plt.plot(t,y_2_hat.real,':', color = color_estimated)

    ax_uv = plt.subplot(gs[:2, :2])
    plt.plot(y_1.real+b_1.real,y_2.real+b_2.real,color = 'black', alpha = alpha_noisy)
    plt.plot(y_1.real,y_2.real, color = color_th)
    plt.plot(y_1_hat.real,y_2_hat.real,':', color = color_estimated)
    
    ax_u_r = {}
    ax_v_r = {}
    ax_uv_r = {}
    
    for r in range(R):
        
        y_1_r, y_2_r = quaternion_to_complex(rdot(M[:,r],q[r]))
        y_1_r_hat, y_2_r_hat = quaternion_to_complex(rdot(M_hat[:,r],q_hat[r]))
        
        ax_u_r[r] = plt.subplot(gs[2+2*r,2:])
        plt.plot(t,y_1_r.real+b_1.real,color = 'black', alpha = alpha_noisy)
        plt.plot(t,y_1_r.real,color = color_th)
        plt.plot(t,y_1_r_hat.real, ':', color = color_estimated)
        
        ax_v_r[r] = plt.subplot(gs[2+2*r+1,2:])
        plt.plot(t,y_2_r.real+b_2.real,color = 'black', alpha = alpha_noisy)
        plt.plot(t,y_2_r.real,color = color_th)
        plt.plot(t,y_2_r_hat.real, ':', color = color_estimated)
        
        ax_uv_r[r] = plt.subplot(gs[(2+2*r):(2+2*r+2),:2])
        plt.plot(y_1_r.real+b_1.real,y_2_r.real+b_2.real,color = 'black', alpha = alpha_noisy)
        plt.plot(y_1_r.real,y_2_r.real, color = color_th)
        plt.plot(y_1_r_hat.real,y_2_r_hat.real, ':', color = color_estimated)
    
    if method_limits=="noise":
        val_lim = np.max(np.abs(ax_u.get_ylim()+ax_v.get_ylim()))
    else:
        val_lim = 1.1*(np.max( np.reshape([ np.abs(y_1.real),np.abs(y_2.real) ],[-1,1]) ))
    
    ax_u.set_xlim([t[0],t[N-1]])
    ax_u.set_ylim([-val_lim,val_lim])
    ax_u.legend(["Noised","Theory","Estimation"], loc = "upper right")
    ax_u.set_ylabel("$u(t)$")
    
    ax_v.set_xlim([t[0],t[N-1]])
    ax_v.set_ylim([-val_lim,val_lim])
    ax_v.legend(["Noised","Theory","Estimation"], loc = "upper right")
    ax_v.set_ylabel("$v(t)$")

    ax_uv.set_xlim([-val_lim,val_lim])
    ax_uv.set_ylim([-val_lim,val_lim])
    ax_uv.legend(["Noised","Theory","Estimation"], loc = "upper right")
    ax_uv.set_ylabel("$v(t)$")
    
    for r in range(R):
        if method_limits=="noise":
            val_lim_r = np.max(np.abs(ax_u_r[r].get_ylim()+ax_v_r[r].get_ylim()))
        elif method_limits=="estimation":
            y_1_r_hat, y_2_r_hat = quaternion_to_complex(rdot(M_hat[:,r],q_hat[r]))
            val_lim_r = 1.1*np.max([np.max(np.abs(y_1_r_hat.real)),np.max(np.abs(y_2_r_hat.real))])
        else:
            y_1_r, y_2_r = quaternion_to_complex(rdot(M[:,r],q[r]))
            val_lim_r = 1.1*np.max([np.max(np.abs(y_1_r.real)),np.max(np.abs(y_2_r.real))])
        
        ax_u_r[r].set_xlim([t[0],t[N-1]])
        ax_u_r[r].set_ylim([-val_lim_r,val_lim_r])
        ax_u_r[r].legend(["Noised","Theory","Estimation"], loc = "upper right")
        ax_u_r[r].set_ylabel("$u(t)$")

        ax_v_r[r].set_xlim([t[0],t[N-1]])
        ax_u_r[r].set_ylim([-val_lim_r,val_lim_r])
        ax_v_r[r].legend(["Noised","Theory","Estimation"], loc = "upper right")
        ax_v_r[r].set_ylabel("$v(t)$")
        if r==(R-1):
            ax_v_r[r].set_xlabel("$t$")
        
        ax_uv_r[r].set_xlim([-val_lim_r,val_lim_r])
        ax_uv_r[r].set_ylim([-val_lim_r,val_lim_r])
        ax_uv_r[r].legend(["Noised","Theory","Estimation"], loc = "upper right")
        ax_uv_r[r].set_ylabel("$v(t)$")
        if r==(R-1):
            ax_uv_r[r].set_xlabel("$u(t)$")

    plt.show()


def plot_blind_estimated_ellipses(t,y,M_hat,q_hat, color_th = "#1f77b4", color_estimated = "#ff7f0e"):
    
    R = np.shape(q_hat)[0]
    N = np.shape(M_hat)[0]
    
    mod_r = np.zeros(R)
    for r in range(R):
        mod_r[r] = -np.mean(abs(rdot(M_hat[:,r],q_hat[r]))**2)
    
    perm_mod = np.argsort(mod_r)
    
    y_1, y_2 = quaternion_to_complex(y)
    
    y_hat = rdot(M_hat,q_hat)
    y_1_hat, y_2_hat = quaternion_to_complex(y_hat)
    
    plt.figure(figsize = [13,4+4*R])
    gs = gridspec.GridSpec(2+2*R, 6)
    gs.update(hspace=0.5, wspace=1, bottom=0.1, left=0.1, top=0.95, right=0.95)

    ax_u = plt.subplot(gs[0, 2:])
    plt.plot(t,y_1.real, color = color_th)
    plt.plot(t,y_1_hat.real,':', color = color_estimated)
    
    ax_v = plt.subplot(gs[1, 2:])
    plt.plot(t,y_2.real, color = color_th)
    plt.plot(t,y_2_hat.real,':', color = color_estimated)

    ax_uv = plt.subplot(gs[:2, :2])
    plt.plot(y_1.real,y_2.real, color = color_th)
    plt.plot(y_1_hat.real,y_2_hat.real,':', color = color_estimated)
    
    ax_u_r = {}
    ax_v_r = {}
    ax_uv_r = {}
    
    for ind_r in range(R):
        r = perm_mod[ind_r]
        
        y_1_r_hat, y_2_r_hat = quaternion_to_complex(rdot(M_hat[:,r],q_hat[r]))
        
        ax_u_r[ind_r] = plt.subplot(gs[2+2*ind_r,2:])
        plt.plot(t,y_1_r_hat.real, color = color_estimated)
        
        ax_v_r[ind_r] = plt.subplot(gs[2+2*ind_r+1,2:])
        plt.plot(t,y_2_r_hat.real, color = color_estimated)
        
        ax_uv_r[ind_r] = plt.subplot(gs[(2+2*ind_r):(2+2*ind_r+2),:2])
        plt.plot(y_1_r_hat.real,y_2_r_hat.real, color = color_estimated)
    
    val_lim = np.max(np.abs(ax_u.get_ylim()+ax_v.get_ylim()))
    for r in range(R):
        
        # val_lim_r = np.max(np.abs(ax_u_r[r].get_ylim()+ax_v_r[r].get_ylim()))
        val_lim_r = val_lim
        
        ax_u_r[r].set_xlim([t[0],t[N-1]])
        ax_u_r[r].set_ylim([-val_lim_r,val_lim_r])
        ax_u_r[r].legend(["Estimation r = %d" %r], loc = "upper right")
        ax_u_r[r].set_ylabel("$u(t)$")

        ax_v_r[r].set_xlim([t[0],t[N-1]])
        ax_v_r[r].set_ylim([-val_lim_r,val_lim_r])
        ax_v_r[r].legend(["Estimation r = %d" %r], loc = "upper right")
        ax_v_r[r].set_ylabel("$v(t)$")
        if r==(R-1):
            ax_v_r[r].set_xlabel("$t$")
        
        ax_uv_r[r].set_xlim([-val_lim_r,val_lim_r])
        ax_uv_r[r].set_ylim([-val_lim_r,val_lim_r])
        ax_uv_r[r].legend(["Estimation r = %d" %r], loc = "upper right")
        ax_uv_r[r].set_ylabel("$v(t)$")
        if r==(R-1):
            ax_uv_r[r].set_xlabel("$u(t)$")

    
    ax_u.set_xlim([t[0],t[N-1]])
    ax_u.set_ylim([-val_lim,val_lim])
    ax_u.legend(["Theory","Estimation"], loc = "upper right")
    ax_u.set_ylabel("$u(t)$")
    
    ax_v.set_xlim([t[0],t[N-1]])
    ax_v.set_ylim([-val_lim,val_lim])
    ax_v.legend(["Theory","Estimation"], loc = "upper right")
    ax_v.set_ylabel("$v(t)$")

    ax_uv.set_xlim([-val_lim,val_lim])
    ax_uv.set_ylim([-val_lim,val_lim])
    ax_uv.legend(["Theory","Estimation"], loc = "upper right")
    ax_uv.set_ylabel("$v(t)$")
    
    plt.show()