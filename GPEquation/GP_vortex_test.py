import sys, copy, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../NumpyTensorTools')))
import numpy_dmrg as dmrg
import numpy as np
import matplotlib.pyplot as plt
import npmps
import plot_utility as pltut
import hamilt.hamilt_sho as sho
import gradient_descent_old as gd
import qtt_tools as qtt
import hamilt.hamilt_angular_momentum as ang
import tci
import time
import pickle
import gradient_descent_GP as gdGP


def psi_sqr (psi):
    psi_op = qtt.MPS_to_MPO (psi)
    psi_op = npmps.conj (psi_op)
    res = npmps.exact_apply_MPO (psi_op, psi)

    #print('psi2 dim, before compression',npmps.MPS_dims(res))
    #res = npmps.svd_compress_MPS (res, cutoff=1e-12)
    #print('psi2 dim, after compression',npmps.MPS_dims(res))

    return res

# psi2 is the initial guess of the result
def fit_psi_sqr (psi, psi2, maxdim, cutoff=1e-16):
    psi_op = qtt.MPS_to_MPO (psi)
    psi_op = npmps.conj (psi_op)
    fit = dmrg.fit_apply_MPO (psi_op, psi, psi2, numCenter=1, nsweep=1, maxdim=maxdim, cutoff=cutoff)
    #psi2_exact = psi_sqr (psi)
    #print_overlap (psi2_exact, fit)
    return fit

def make_H_GP (H0, psi, psi2, g, maxdim):
    #t1 = time.time()                                # time
   # psi2 = psi_sqr (psi)
    psi2 = fit_psi_sqr(psi, psi2, maxdim = maxdim, cutoff = cutoff)
   # for i in range(len(psi2)):
   #     print(psi2[i].shape)
   # t2 = time.time()                                # time
   # print('psi2 time',(t2-t1))                      # time

    H_psi = qtt.MPS_to_MPO (psi2)
    H_psi[0] *= g
    H = npmps.sum_2MPO (H0, H_psi)
    H = npmps.svd_compress_MPO (H, cutoff=1e-12)
    return H, psi2

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
        enss.append(np.real(en))
        #print('TDVP',n,en)
        #print("inner E = ",np.abs(npmps.inner_MPO (psi, psi, H)))
        t2 = time.time()                                # time
        print('imag time evol time',(t2-t1))                      # time
        t22 = time.time()
        ts.append(t22-t11)

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
        ts.append(t22-t11)
    return psi, enss, ts

def gradient_descent2 (H0, psi, g, step_size, steps, maxdim=100000000, cutoff_mps2 = 1e-6,  cutoff_mps=1e-16):
    psi, enss, ts = gdGP.gradient_descent_GP_MPS (steps, psi, H0, g, step_size, niter=3, maxdim=maxdim, cutoff_mps2=cutoff_mps2, cutoff_mps = cutoff_mps, linesearch=True)
    #en *= dx
    return psi, enss, ts

def gradient_descent3 (H0, psi, g, step_size, steps, maxdim=100000000, cutoff=1e-16):
    enss = []
    ts = []
    t1 = time.time()
    for n in range(steps):
        # Gradient descent
        psi, en = gdGP.gradient_descent_GP_MPS (psi, H0, g, step_size, niter=5, maxdim=maxdim, cutoff=cutoff, linesearch=True)
        #en *= dx
        enss.append(en)
       # print('GD3',n,en)
        t2 = time.time()
        ts.append(t2-t1)
    return psi, enss, ts

def fitfun_xy (x, y):
    f = np.sqrt(1/np.pi)*np.exp(-(x**2+y**2)/2)*np.exp(2*np.pi*1j*np.random.rand())
    return f

def inds_to_num (inds, dx, shift):
    bstr = ''
    for i in inds:
        bstr += str(i)
    return pltut.bin_to_dec (bstr, dx, shift)

def fitfun (inds):
    N = len(inds)//2
    Ndx = 2**N
    dx = (x2-x1)/Ndx
    shift = x1

    xinds, yinds = inds[:N], inds[N:]
    x = inds_to_num (xinds, dx, shift)
    y = inds_to_num (yinds, dx, shift)
    return fitfun_xy (x, y)

def get_init_state (N, x1, x2, maxdim):
    seed = 13
    np.random.seed(seed)
    mps = tci.tci (fitfun, 2*N, 2, maxdim=maxdim)
    mps = npmps.normalize_MPS (mps)
    return mps    

def check_hermitian (mpo):
    mm = npmps.MPO_to_matrix (mpo)
    t = np.linalg.norm(mm - mm.conj().T)
    print(t)
    assert t < 1e-10

def check_the_same (mpo1, mpo2):
    m1 = npmps.MPO_to_matrix(mpo1)
    m2 = npmps.MPO_to_matrix(mpo2)
    d = np.linalg.norm(m1-m2)
    print(d)
    assert d < 1e-10

def print_overlap (mps1, mps2):
    mps1 = copy.copy(mps1)
    mps2 = copy.copy(mps2)
    mps1 = npmps.normalize_MPS(mps1)
    mps2 = npmps.normalize_MPS(mps2)
    print('overlap',npmps.inner_MPS(mps1, mps2))

if __name__ == '__main__':    
    N = 8
    x1,x2 = -6,6
    Ndx = 2**N
    dx = (x2-x1)/Ndx
    print('dx',dx)

    g = 100/dx**2


    maxdim = 16
    cutoff = 1e-12
    krylovDim = 10

    H_SHO = sho.make_H (N, x1, x2)
    H_SHO = npmps.get_H_2D (H_SHO)
    #H_SHO = npmps.change_dtype(H_SHO, complex)
    H_SHO[0] = 0.5*H_SHO[0]

    H0 = H_SHO

    print('Non-interacting MPO dim, before compression:',npmps.MPO_dims(H0))
    H0 = npmps.svd_compress_MPO (H0, cutoff=1e-12)
    print('Non-interacting MPO dim:',npmps.MPO_dims(H0))


    def absSqr (a):
        return abs(a)**2
    absSqr = np.vectorize(absSqr)

    # Initial MPS
    psi = get_init_state (N, x1, x2, maxdim=maxdim)
    print('Initial psi dim, before compression:',npmps.MPS_dims(psi))
    psi = npmps.svd_compress_MPS (psi, cutoff=1e-12)
    print('Initial psi dim:',npmps.MPS_dims(psi))
    #psi = qtt.normalize_MPS_by_integral (psi, x1, x2, Dim=2)

    # TDVP
    dt = dx**2/2
    print('dt',dt)
    psi_TDVP, ens_TDVP, ts0 = imag_time_evol (H0, psi, g, dt, steps=0, maxdim=maxdim, cutoff=cutoff, krylovDim=krylovDim)
    TDVP_CPUTIME = np.column_stack((ts0, ens_TDVP))
    np.savetxt('TDVP_CPUTIME.txt', TDVP_CPUTIME, fmt='%.12f')
    with open('7_vortex_tdvp.pkl', 'wb') as f:
        pickle.dump(psi_TDVP, f)

    # Gradient descent
    step_size = dt*0.01
    psi_GD, ens_GD, ts1 = gradient_descent (H0, psi, g, step_size, steps=0, maxdim=maxdim, cutoff=cutoff)
    GD_CPUTIME = np.column_stack((ts1, ens_GD))
    np.savetxt('GD_CPUTIME.txt', GD_CPUTIME, fmt='%.12f')
    # Gradient descent
    psi_GD2, ens_GD2, ts2 = gradient_descent2 (H0, psi, g, step_size=dt, steps=10000, maxdim=maxdim, cutoff_mps2=1e-8, cutoff_mps = cutoff)
    GD2_CPUTIME = np.column_stack((ts2, ens_GD2))
    np.savetxt('GD2_CPUTIME.txt', GD2_CPUTIME, fmt='%.12f')
    with open('7_vortex_gd.pkl', 'wb') as f:
        pickle.dump(psi_GD2, f)


    # Gradient descent
    #psi_GD3, ens_GD3, ts3 = gradient_descent3 (H0, psi, g, step_size=1, steps=10, maxdim=maxdim, cutoff=cutoff)



    '''# Grow site
    for i in range(1,2):
        dx *= 0.5
        g *= 2
        gamma *= 0.1
        H02 = sho.make_H (N+i, x1, x2)
        print(len(H02))
        H02 = npmps.get_H_2D (H02)
        psi_GD2 = qtt.grow_site_2D (psi_GD)


        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        pltut.plot_2D (psi_GD2, x1, x2, ax=ax, title='Init')
        plt.show()

        print(len(H02),len(psi_GD2))
        psi_GD2, ens_GD2 = gradient_descent (H02, psi_GD2, g, gamma, steps=100)'''


    # Plot energy
    fig2, ax2 = plt.subplots()
    ax2.plot(range(len(ens_TDVP)), ens_TDVP, label='TDVP step')
    ax2.plot(range(len(ens_GD2)), ens_GD2, label='GD2 step')
    ax2.set_xlabel(r"time step", loc="center")
    ax2.set_ylabel(r"$<\mu(step)>$", loc="center")
    ax2.set_yscale('log')
    ax2.legend()

    #en = ens_GD2[-1]        # replace by the exact energy
    fig2, ax2 = plt.subplots()
    ax2.plot(ts0, ens_TDVP, marker='.', label='TDVP Wall time')
    ax2.plot(ts2, ens_GD2, marker='+', label='GD2 Wall time')
    #ax2.plot(ts0, abs(ens_TDVP-en), marker='.', label='TDVP')
    #ax2.plot(ts2, abs(ens_GD2-en), marker='+', label='GD2')
    ax2.set_xlabel(r"Wall time(s)", loc="center")
    ax2.set_ylabel(r"$<\mu(t)>$", loc="center")
    ax2.set_yscale('log')
    ax2.legend()
    #plt.show()

    # Plot wavefunction
    psi = qtt.normalize_MPS_by_integral (psi, x1, x2, Dim=2)
    psi2 = psi_sqr (psi)
    X, Y, Z = pltut.plot_2D (psi2, x1, x2, ax=None, func=absSqr, label='Init')
    #fig.savefig('init.pdf')
    # 
    '''fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    psi_TDVP = qtt.normalize_MPS_by_integral (psi_TDVP, x1, x2, Dim=2)
    psi_TDVP2 = psi_sqr (psi_TDVP)
    X_TDVP, Y_TDVP, Z_TDVP = pltut.plot_2D (psi_TDVP, x1, x2, ax=ax, func=absSqr, label='TDVP')
    #ax.legend()
    fig.savefig('TDVP.pdf')

    #
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    psi_GD = qtt.normalize_MPS_by_integral (psi_GD, x1, x2, Dim=2)
    psi_GD2 = psi_sqr (psi_GD)
    X_GD, Y_GD, Z_GD = pltut.plot_2D (psi_GD, x1, x2, ax=ax, func=absSqr, label='GD')
    fig.savefig('GD.pdf')
    #ax.legend()'''

    #plot the contour of the "|psi(x,y)|**2 via the Gradient descent method with line search"
    psi_GD2 = qtt.normalize_MPS_by_integral (psi_GD2, x1, x2, Dim=2)
    pltut.plot_2D_proj (psi_GD2, x1, x2, ax=None, func=absSqr, label='GD2_proj')

    #plot the contour of the "|psi(x,y)|**2 via the TDVP method"
    psi_TDVP = qtt.normalize_MPS_by_integral (psi_TDVP, x1, x2, Dim=2)

    pltut.plot_2D_proj (psi_TDVP, x1, x2, ax=None, func=absSqr, label='TDVP_proj')


    '''fig, ax = plt.subplots()
    y = 2**N//2
    Z = absSqr(Z)
    Z_TDVP = absSqr(Z_TDVP)
    Z_GD = absSqr(Z_TDVP)
    ax.plot (X[y,:], Z[y,:], label='Init')
    ax.plot (X_TDVP[y,:], Z_TDVP[y,:], label='TDVP')
    ax.plot (X_GD[y,:], Z_GD[y,:], label='GD')
    ax.legend()'''

    plt.show()
