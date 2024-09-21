import sys, copy
import npmps
from ncon import ncon
import numpy_dmrg as dmrg
import numpy as np
import time
import matplotlib.pyplot as plt
sys.path.append('/home/chiamin/NumpyTensorTools/gradient_descent/')
import gradient_descent as gd
import qtt_tools as qtt

# Function to be optimized
class cost_function:
    def __init__ (self, L, M, R):
        self.L = L
        self.R = R
        self.M = M
        self.effH = dmrg.eff_Hamilt_1site (L, M, R)

    def gradient (self, x):
        return 2 * self.effH.apply (x)

    def val_slope (self, x):
        grad = self.gradient (x)
        value = 0.5 * self.inner (grad, x).real
        return value, grad

    def grad_x (self, grad, x):
        return self.inner (grad, x)

    # ------ Must be defined if we require ||x||=1 ------
    def norm (self, x):
        return np.linalg.norm(x)

    def inner (self, x1, x2):
        return np.inner(x1.flatten().conj(), x2.flatten())
    # ---------------------------------------------------

def get_en (psi, H):
    en0 = npmps.inner_MPO (psi, psi, H)
    return en0


def gradient_descent_MPS (mps, mpo, step_size, niter=1, maxdim=100000000, cutoff=1e-16, linesearch=False):
    assert len(mps) == len(mpo)
    npmps.check_MPO_links (mpo)
    npmps.check_MPS_links (mps)
    #npmps.check_canonical (mps, 0)

    mps = copy.copy(mps)

    N = len(mps)
    LR = dmrg.LR_envir_tensors_mpo (N)

    sites = [range(N-1), range(N-1,-1,-1)]
    start = time.time()
    for lr in [0,1]:
        for p in sites[lr]:
            for i in range(niter):

                LR.update_LR (mps, mps, mpo, p)
                func = cost_function (LR[p-1], mpo[p], LR[p+1])



                #gg, en = func.grad_val (mps[p])
                #een = get_en (mps, mpo)
                #print('ggg',en,een)




                mps[p], grad, en = gd.gradient_descent (func, mps[p], step_size=step_size, normalize=True, linesearch=linesearch)

            npmps.check_MPS_links (mps)
            mps = dmrg.orthogonalize_MPS_tensor (mps, p, toRight=(lr==0), maxdim=maxdim, cutoff=cutoff)

    end = time.time()
    gd_time = end - start

    print('GD gradient time',gd_time)

    return mps, en

