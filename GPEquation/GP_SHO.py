import sys, copy, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../NumpyTensorTools')))
import numpy_dmrg as dmrg
import numpy as np
import matplotlib.pyplot as plt
import polynomial as poly
import differential as diff
import npmps
import plot_utility as ptut
import hamilt.hamilt_sho as sho
import gradient_descent_old as gd
import qtt_tools as qtt
import time

def psi_sqr (psi):
    psi_op = qtt.MPS_to_MPO (psi)
    res = npmps.exact_apply_MPO (psi_op, psi)
    res = npmps.svd_compress_MPS (res, cutoff=1e-12)
    return res

# psi2 is the initial guess of the result
def fit_psi_sqr (psi, psi2):
    psi_op = npmps.MPS_to_MPO (psi)
    fit, overlap = dmrg.fit_apply_MPO (psi_op, psi, psi2, numCenter=1, nsweep=1, cutoff=1e-12)
    return fit

def make_H_GP (H0, psi, psi2, g, maxdim):
    t1 = time.time()                                # time
    psi2 = psi_sqr (psi)
    t2 = time.time()                                # time
    #print('psi2 time',(t2-t1))                      # time

    H_psi = qtt.MPS_to_MPO (psi2)
    H_psi[0] *= g
    H = npmps.sum_2MPO (H0, H_psi)
    H = npmps.svd_compress_MPO (H, cutoff=1e-12)
    return H, psi2

# def imag_time_evol (H0, psi, g, dt, steps, maxdim, cutoff=1e-16):
#     psi = copy.copy(psi)
#     psi2 = psi_sqr (psi)
#     enss = []
#     for n in range(steps):
#         # Update the Hamiltonian
#         H, psi2 = make_H_GP (H0, psi, psi2, g, maxdim)
#         # TDVP
#         psi, ens, terrs = dmrg.tdvp (1, psi, H, dt, [maxdim], cutoff=cutoff, krylovDim=20, verbose=False)
#         en = ens[-1]
#         enss.append(en)
#         print('TDVP',n,en)

#     return psi, enss

def imag_time_evol (H0, psi, g, dt, steps, maxdim, cutoff=1e-12, krylovDim=10):
    psi = copy.copy(psi)
    psi2 = psi_sqr (psi)
    enss = []
    ts = []
    t11 = time.time()
    for n in range(steps):
        t1 = time.time()                                # timedx
        # Update the Hamiltonian
        H, psi2 = make_H_GP (H0, psi, psi2, g, maxdim=maxdim)

        # TDVP
        psi, ens, terrs = dmrg.tdvp (1, psi, H, dt, [maxdim], cutoff=cutoff, krylovDim=krylovDim, verbose=False)
        en = np.abs(ens[-1])
        enss.append(en)
        #print('TDVP',n,en)
        t2 = time.time()                                # time
        #print('imag time evol time',(t2-t1))                      # time
        t22 = time.time()
        #ts.append(t22-t11)
        ts.append(t2-t1)


    return psi, enss, ts

def gradient_descent (H0, psi, g, gamma, steps, maxdim, cutoff):
    psi = copy.copy(psi)
    psi2 = psi_sqr (psi)
    enss = []
    ts = []
    t11 = time.time()
    for n in range(steps):
        t1 = time.time()                                # time
        # Update the Hamiltonian
        H, psi2 = make_H_GP (H0, psi, psi2, g, maxdim)

        # Gradient descent
        psi, en = gd.gradient_descent (psi, H, gamma, maxdim=maxdim, cutoff=cutoff)
        #en *= dx
        t2 = time.time()
        enss.append(np.abs(en))
        #print('GD',n,en)
        #print('GD time',(t2-t1))                      # time
        t22 = time.time()
        #ts.append(t22-t11)
        ts.append(t2-t1)
    return psi, enss, ts

if __name__ == '__main__':
    N = 8
    x1,x2 = -15,15

    Ndx = 2**N
    dx = (x2-x1)/Ndx
    print('dx',dx)

    g = 62.742/dx

    maxdim = 10
    cutoff = 1e-12

    H0 = sho.make_H (N, x1, x2)
    H0[0] = 0.5* H0[0]

    # Initial MPS
    psi = npmps.random_MPS (N,2,maxdim)
    psi = npmps.to_canonical_form (psi, 0)
    psi[0] /= np.linalg.norm(psi[0])

    psi, ens, terrs = dmrg.dmrg (1, psi, H0, [10]*10, cutoff=1e-12, krylovDim=10, verbose=True)


    # TDVP
    dt = dx**2
    print('dt',dt)
    psi_TDVP, ens_TDVP, ts0 = imag_time_evol (H0, psi, g, dt, steps=0, maxdim=maxdim, cutoff=cutoff,krylovDim=10)
    TDVP_CPUTIME = np.column_stack((ts0, ens_TDVP))
    np.savetxt('TDVP_1D_CPUTIME.txt', TDVP_CPUTIME, fmt='%f')
    print('TDVP_TIME',np.sum(ts0)/500)
    # Gradient descent
    step_size = dt*0.01
    psi_GD, ens_GD, ts1 = gradient_descent (H0, psi, g, step_size, steps=500, maxdim=maxdim, cutoff=cutoff)
    print('GD_TIME',np.sum(ts1)/500)
    GD_CPUTIME = np.column_stack((ts1, ens_GD))
    np.savetxt('GD_1D_CPUTIME.txt', GD_CPUTIME, fmt='%f')
    # Grow site
    '''for i in range(1,2):
        dx *= 0.5
        g *= 2
        gamma *= 0.1
        H02 = sho.make_H (N+i, x1, x2)
        psi_GD2 = qtt.grow_site_1D (psi_GD)
        psi_GD2, ens_GD2 = gradient_descent (H02, psi_GD2, g, gamma, steps=100)'''


    # Plot energy
    fig2, ax2 = plt.subplots()
    ax2.plot(range(len(ens_TDVP)), ens_TDVP, label='TDVP')
    ax2.plot(range(len(ens_GD)), ens_GD, label='GD')
    #ax2.plot(range(len(ens_GD2)), ens_GD, label='GD2')
    ax2.legend()


    # Plot wavefunction
    fig, ax = plt.subplots()
    # initial
    sho.plot_GS_exact(x1,x2,ax,ls='--',label='Exact')
    ptut.plot_1D (psi, x1, x2, ax=ax, label='Init')
    # exact
    #dat = np.loadtxt('twotdvpD=10Gaussfinalpsi.txt')
    #ax.plot(dat[:,0],dat[:,1],ls='--',label='exact')
    # 
    psi_TDVP = qtt.normalize_MPS_by_integral (psi_TDVP, x1, x2, Dim=1)
    psi_GD = qtt.normalize_MPS_by_integral (psi_GD, x1, x2, Dim=1)
    ptut.plot_1D (psi_TDVP, x1, x2, ax=ax, func=None, label='TDVP')
    ptut.plot_1D (psi_GD, x1, x2, ax=ax, func=None, label='GD')
    #ptut.plot_1D (psi_GD2, x1, x2, ax=ax, label='GD2')
    ax.legend()


    plt.show()
