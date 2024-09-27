import sys, copy
sys.path.append('/home/chiamin/NumpyTensorTools/gradient_descent/')
sys.path.append('/home/chiamin/NumpyTensorTools/')
import npmps
from ncon import ncon
import numpy_dmrg as dmrg
import numpy as np
import time
import gradient_descent as gd
import qtt_tools as qtt
import matplotlib.pyplot as plt
import cost_function_GP_new as costf_new
import MPS_square as mpssqr

def left_environment_gradient_dag (L, A):
    W = qtt.MPS_tensor_to_MPO_tensor (A)
    #  ----
    #  |  |---------O--- -1
    #  |  |    1    |
    #  |  |        -2
    #  |  |
    #  |  |--- -3
    #  |  |
    #  |  |--- -4
    #  |  |
    #  |  |--- -5
    #  ----
    tmp = ncon((L, np.conj(A)),((1,-3,-4,-5),(1,-2,-1)))
    #  ----
    #  |  |---------O--- -1
    #  |  |         |
    #  |  |       1 |
    #  |  |---------O--- -2
    #  |  |    2    |
    #  |  |        -3
    #  |  |
    #  |  |--- -4
    #  |  |
    #  |  |--- -5
    #  ----
    tmp = ncon((tmp,np.conj(W)),((-1,1,2,-4,-5),(2,1,-3,-2)))
    #  ----
    #  |  |---------O--- -1
    #  |  |         |
    #  |  |         |
    #  |  |---------O--- -2
    #  |  |         |
    #  |  |       1 |
    #  |  |---------O--- -3
    #  |  |    2    |
    #  |  |        -4
    #  |  |
    #  |  |--- -5
    #  ----
    tmp = ncon((tmp,W),((-1,-2,1,2,-5),(2,1,-4,-3)))
    return tmp

def right_environment_gradient_dag (R, A):
    W = qtt.MPS_tensor_to_MPO_tensor (A)
    #                 ----
    #  -1 ---O--------|  |
    #        |   1    |  |
    #       -2        |  |
    #           -3 ---|  |
    #                 |  |
    #           -4 ---|  |
    #                 |  |
    #           -5 ---|  |
    #                 ----
    tmp = ncon((R,np.conj(A)),((1,-3,-4,-5),(-1,-2,1)))
    #                 ----
    #  -1 ---O--------|  |
    #        |        |  |
    #        | 1      |  |
    #  -2 ---O--------|  |
    #        |   2    |  |
    #       -3        |  |
    #           -4 ---|  |
    #                 |  |
    #           -5 ---|  |
    #                 ----
    tmp = ncon((tmp,np.conj(W)),((-1,1,2,-4,-5),(-2,1,-3,2)))
    #                 ----
    #  -1 ---O--------|  |
    #        |        |  |
    #        |        |  |
    #  -2 ---O--------|  |
    #        |        |  |
    #        | 1      |  |
    #  -3 ---O--------|  |
    #        |   2    |  |
    #       -4        |  |
    #                 |  |
    #           -5 ---|  |
    #                 ----
    tmp = ncon((tmp,W),((-1,-2,1,2,-5),(-3,1,-4,2)))
    return tmp

def left_environment_gradient (L, A):
    W = qtt.MPS_tensor_to_MPO_tensor (A)
    #  ----
    #  |  |--- -1
    #  |  |
    #  |  |--- -2
    #  |  |
    #  |  |--- -3
    #  |  |
    #  |  |        -4
    #  |  |    1    |
    #  |  |---------O--- -5
    #  ----
    tmp = ncon((L, A),((-1,-2,-3,1),(1,-4,-5)))
    #  ----
    #  |  |--- -1
    #  |  |
    #  |  |--- -2
    #  |  |
    #  |  |        -3
    #  |  |    1    |
    #  |  |---------O--- -4
    #  |  |       2 |
    #  |  |         |
    #  |  |---------O--- -5
    #  ----
    tmp = ncon((tmp,W),((-1,-2,1,2,-5),(1,-3,2,-4)))
    #  ----
    #  |  |--- -1
    #  |  |
    #  |  |        -2
    #  |  |    1    |
    #  |  |---------O--- -3
    #  |  |       2 |
    #  |  |         |
    #  |  |---------O--- -4
    #  |  |         |
    #  |  |         |
    #  |  |---------O--- -5
    #  ----
    tmp = ncon((tmp,np.conj(W)),((-1,1,2,-4,-5),(1,-2,2,-3)))
    return tmp

def right_environment_gradient (R, A):
    W = qtt.MPS_tensor_to_MPO_tensor (A)
    #                 ----
    #           -1 ---|  |
    #                 |  |
    #           -2 ---|  |
    #                 |  |
    #           -3 ---|  |
    #                 |  |
    #       -4        |  |
    #        |   1    |  |
    #  -5 ---O--------|  |
    #                 ----
    tmp = ncon((R,A),((-1,-2,-3,1),(-5,-4,1)))
    #                 ----
    #           -1 ---|  |
    #                 |  |
    #           -2 ---|  |
    #                 |  |
    #       -3        |  |
    #        |   1    |  |
    #  -4 ---O--------|  |
    #        | 2      |  |
    #        |        |  |
    #  -5 ---O--------|  |
    #                 ----
    tmp = ncon((tmp,W),((-1,-2,1,2,-5),(-4,-3,2,1)))
    #                 ----
    #           -1 ---|  |
    #                 |  |
    #       -2        |  |
    #        |   1    |  |
    #  -3 ---O--------|  |
    #        | 2      |  |
    #        |        |  |
    #  -4 ---O--------|  |
    #        |        |  |
    #        |        |  |
    #  -5 ---O--------|  |
    #                 ----
    tmp = ncon((tmp,np.conj(W)),((-1,1,2,-4,-5),(-3,-2,2,1)))
    return tmp

class LR_envir_tensors_phi4:
    def __init__ (self, N, dtype=float):
        self.dtype = dtype
        self.centerL = 0
        self.centerR = N-1
        self.LR = dict()
        for i in range(-1,N+1):
            self.LR[i] = None
        self.LR[-1] = np.ones((1,1,1,1),dtype=dtype)
        self.LR[N] = np.ones((1,1,1,1),dtype=dtype)

    def __getitem__(self, i):
        if i >= self.centerL and i <= self.centerR:
            print('environment tensor is not updated')
            print('centerL,centerR,i =',self.centerL,self.centerR,i)
            raise Exception
        return self.LR[i]

    def delete (self, i):
        self.centerL = min(self.centerL, i)
        self.centerR = max(self.centerR, i)

    def get_centers (self):
        return self.centerL, self.centerR

    # self.LR[i-1] and self.LR[i+1] are the left and right environments for the ith site
    # self.LR[i] for i=-1,...,centerL-1 are left environments;
    # self.LR[i] for i=centerR+1,...,N are right environments
    # MPS tensor indices = (l,i,r)
    # MPO tensor indices = (l,ip,i,r)
    # Left or right environment tensor = (up, mid, down)
    def update_LR (self, mps, centerL, centerR=None):
        # Set dtype
        dtype = mps[0].dtype
        if dtype != self.dtype:
            for i in self.LR:
                if self.LR[i] != None:
                    self.LR[i] = self.LR[i].astype(dtype)
            self.dtype = dtype

        if centerR == None:
            centerR = centerL
        if centerL > centerR+1:
            print('centerL cannot be larger than centerR+1')
            print('centerL, centerR =',centerL, centerR)
            raise Exception
        # Update the left environments
        for p in range(self.centerL, centerL):
            A = mps[p]
            #  ----
            #  |  |--- -1
            #  |  |
            #  |  |        -2
            #  |  |         |
            #  |  |---------O--- -3
            #  |  |         |
            #  |  |         |
            #  |  |---------O--- -4
            #  |  |         |
            #  |  |         |
            #  |  |---------O--- -5
            #  ----
            tmp = left_environment_gradient (self.LR[p-1], A)
            #  ----     1
            #  |  |---------O--- -1
            #  |  |         | 2
            #  |  |         |
            #  |  |---------O--- -2
            #  |  |         |
            #  |  |         |
            #  |  |---------O--- -3
            #  |  |         |
            #  |  |         |
            #  |  |---------O--- -4
            #  ----
            self.LR[p] = ncon((tmp,np.conj(A)),((1,2,-2,-3,-4),(1,2,-1)))
        # Update the right environments
        for p in range(self.centerR, centerR, -1):
            A = mps[p]
            #                 ----
            #           -1 ---|  |
            #                 |  |
            #       -2        |  |
            #        |        |  |
            #  -3 ---O--------|  |
            #        |        |  |
            #        |        |  |
            #  -4 ---O--------|  |
            #        |        |  |
            #        |        |  |
            #  -5 ---O--------|  |
            #                 ----
            tmp = right_environment_gradient (self.LR[p+1], A)
            #             1   ----
            #  -1 ---O--------|  |
            #        | 2      |  |
            #        |        |  |
            #  -2 ---O--------|  |
            #        |        |  |
            #        |        |  |
            #  -3 ---O--------|  |
            #        |        |  |
            #        |        |  |
            #  -4 ---O--------|  |
            #                 ----
            self.LR[p] = ncon((tmp,np.conj(A)),((1,2,-2,-3,-4),(-1,2,1)))

        self.centerL = centerL
        self.centerR = centerR

def gradient_descent_GP (func, x, step_size, linesearch=False, normalize=True):
    if normalize:
        assert abs(np.linalg.norm (x) - 1) < 1e-12

    grad = func.env0

    global t_setd
    t1 = time.time()

    direction = -grad
    func.set_direction (direction)


    '''_, slope = func.val_slope (0)
    if abs(slope) < 1e-12:
        print('small slope',slope)
        return x, grad'''

    if linesearch:
        step_size = gd.line_search (func, step_size=step_size, c1=1e-4, c2=0.9)

    x_next = x + step_size * direction
    if normalize:
        x_next = x_next / np.linalg.norm (x_next)

    return x_next, grad, step_size

# mpo is the non-interacting Hamiltonian
def gradient_descent_GP_MPS (nsweep, mps, mpo, g, step_size, niter=1, maxdim=100000000, cutoff_mps2 = 1e-16, cutoff_mps = 1e-12 , linesearch=False):
    assert len(mps) == len(mpo)
    npmps.check_MPO_links (mpo)
    npmps.check_MPS_links (mps)
    #npmps.check_canonical (mps, 0)

    mps = copy.copy(mps)


    N = len(mps)
    LR = dmrg.LR_envir_tensors_mpo (N)
    #LR4 = LR_envir_tensors_phi4 (N)
    LR4 = mpssqr.MPSSquare (N, cutoff=cutoff_mps2)    
    LR4.update_LR (mps, 0)

    '''LR4_2 = mpssqr.MPSSquare (N, cutoff=cutoff, conj=True)
    LR4_2.update_LR (mps, 0)

    def print_dim (psi2):
        for i in range(N):
            print(psi2.dim(i),end=' ')
        print()
    print_dim (LR4)
    print_dim (LR4_2)
    print('-----------------')
    LR4.move_center(mps, len(mps)-1, cutoff=1e-12)
    LR4_2.move_center(mps, len(mps)-1, cutoff=1e-12)
    print_dim (LR4)
    print_dim (LR4_2)
    exit()'''

    sites = [range(N-1), range(N-1,-1,-1)]
    ens = []
    ts = []
    t1 = time.time()
    psi2_dim = []
    for s in range(nsweep):
        LR4dim = []
        for i in range(N):
    	    LR4dim.append(LR4.dim(i))
        max_dim_psi2 = np.max(LR4dim)
        psi2_dim.append(max_dim_psi2)
        for lr in [0,1]:
            for p in sites[lr]:

                LR.update_LR (mps, mps, mpo, p)
                LR4.update_LR (mps, p)
                #LR4.move_center (mps, p, maxdim=maxdim, cutoff=cutoff)

                for n in range(niter):
                    func2 = costf_new.cost_function_GP (LR[p-1], mpo[p], LR[p+1], LR4[p-1], mps[p], LR4[p+1], g)

                    A = mps[p]
                    mps[p], grad, step_size = gradient_descent_GP (func2, mps[p], step_size=step_size, linesearch=linesearch)
                    if step_size < 1e-6:
                        step_size = 1e-6
                    en = np.inner (grad.conj().flatten(), A.flatten())

                    func2.move (step_size)

                mps = dmrg.orthogonalize_MPS_tensor (mps, p, toRight=(lr==0), maxdim=maxdim, cutoff=cutoff_mps)

                #print_dim (LR4)

        ens.append(en)
        t2 = time.time()
        ts.append(t2-t1)
        print('sweep',s,'en =',en)
    np.savetxt('GD2_psi2_dim.txt', psi2_dim, fmt='%d')
    return mps, ens, ts
