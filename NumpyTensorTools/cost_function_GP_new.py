import sys, copy
#sys.path.append('/home/chiamin/NumpyTensorTools/gradient_descent/')
#sys.path.append('/home/chiamin/NumpyTensorTools/')
import npmps
from ncon import ncon
import numpy_dmrg as dmrg
import numpy as np
import qtt_tools as qtt


def inner (A1, A2):
    return np.inner(A1.conj().flatten(), A2.flatten())

def psi2_contraction (L, R, A1, A2):
    #        ----
    #        |  |--- -2
    #        |  |
    #  -1 ---|  |       -3
    #        |  |   1    |
    #        |  |--------O--- -4
    #        ----
    tmp = ncon((L,A1),((-1,-2,1),(1,-3,-4)))
    #                   -2
    #        ----        |
    #        |  |--------O--- -3
    #        |  |   1    |
    #  -1 ---|  |        | 2
    #        |  |        |
    #        |  |--------O--- -4
    #        ----
    W = qtt.MPS_tensor_to_MPO_tensor (A2)
    tmp = ncon((tmp,W),((-1,1,2,-4),(1,-2,2,-3)))
    #
    #                   -2        
    #        ----        |    1   ----
    #        |  |--------O--------|  |
    #        |  |        |        |  |
    #  -1 ---|  |        |        |  |--- -3
    #        |  |        |    2   |  |
    #        |  |--------O--------|  |
    #        ----                 ----
    tmp = ncon((tmp,R),((-1,-2,1,2),(1,2,-3)))
    return tmp

def psi4_contraction (L, R, A1, A2, A3, A4):
    AA1 = psi2_contraction (L, R, A1, A2)
    AA2 = psi2_contraction (L, R, A2, A3)
    return inner(AA2,AA1)

#            ----                           ----
#            |  |--- -2               -1 ---|  |
#            |  |                           |  |
#  L = -1 ---|  |                 R =       |  |--- -3
#            |  |                           |  |
#            |  |--- -3               -2 ---|  |
#            ----                           ----
#
#
#               -2                         -2                         -2
#                |                          |                          |
#  Apsi2 = -1 ---O--- -3          A = -1 ---O--- -3    ==>   W = -1 ---O--- -4
#                                                                      |
#                                                                     -3
def psi4_environment (L, R, Apsi2, A):
    W = qtt.MPS_tensor_to_MPO_tensor (A)
    #        ----
    #        |  |--- -2
    #        |  |
    #  -1 ---|  |
    #        |  |        -3
    #        |  |    1    |
    #        |  |---------O--- -5
    #        ----         |
    #                    -4
    tmp = ncon((L,W),((-1,-2,1),(1,-3,-4,-5)))
    #        ----
    #        |  |--- -1
    #        |  |
    #   -----|  |
    #   |    |  |        -2
    #   |    |  |         |
    #   |    |  |---------O--- -3
    #   |    ----         |
    #   |                 | 2
    #   |            1    |
    #   ------------------O--- -4
    tmp = ncon((tmp.conj(),Apsi2),((1,-1,-2,2,-3),(1,2,-4)))
    #        ----                   ----
    #        |  |--- -1       -3 ---|  |
    #        |  |                   |  |
    #   -----|  |                   |  |-----
    #   |    |  |        -2         |  |    |
    #   |    |  |         |         |  |    |
    #   |    |  |---------O---------|  |    |
    #   |    ----         |    1    ----    |
    #   |                 |                 |
    #   |                 |    2            |
    #   ------------------O------------------
    tmp = ncon((tmp,R.conj()),((-1,-2,1,2),(-3,1,2)))
    return tmp

# Quartic term
class cost_function_phi4:
    def __init__ (self, L, R, x, normalize=True):
        self.L = L
        self.R = R
        self.x = x
        self.normalize = normalize
        if normalize:
            assert abs(1-np.linalg.norm(x)) < 1e-12
        self.psi2xx0 = psi2_contraction (L, R, x, x)
        self.val0 = inner (self.psi2xx0, self.psi2xx0).real

        # df/dA
        self.env0 = psi4_environment (L, R, self.psi2xx0, x)

    def set_direction (self, d):
        self.d = d
        d_new = d
        if self.normalize:
            xd = inner (self.x, d)               # complex number
            d_new = d - xd * self.x                  # complex vector

        psi2xd0 = psi2_contraction (self.L, self.R, self.x, d_new)
        self.slope0 = 4 * inner (psi2xd0, self.psi2xx0).real

    def val_slope (self, a):
        if a == 0:
            return self.val0, self.slope0

        x = self.x + a * self.d
        if self.normalize:
            norm = np.linalg.norm(x)
            x = x / norm

        phi2xx = psi2_contraction(self.L, self.R, x, x)
        val = inner(phi2xx,phi2xx).real

        d = self.d
        # rescale d
        if self.normalize:
            d = d / norm
            xd = inner (x, d)               # complex number
            d = d - xd * x                  # complex vector

        phi2xd = psi2_contraction(self.L, self.R, x, d)
        slope = 4 * inner(phi2xd, phi2xx).real             # real number
        return val, slope

    def move (self, a):
        self.x += a * self.d

        if self.normalize:
            self.norm0 = np.linalg.norm(self.x)
            self.x = self.x / self.norm0
        self.psi2xx0 = psi2_contraction (self.L, self.R, self.x, self.x)
        self.val0 = inner (self.psi2xx0, self.psi2xx0)
        self.env0 = psi4_environment (self.L, self.R, self.psi2xx0, self.x)
        self.d = None

class cost_function_xHx:
    def __init__ (self, L, M, R, x, normalize=True):
        self.x = x
        self.effH = dmrg.eff_Hamilt_1site (L, M, R)

        self.normalize = normalize
        if normalize:
            assert abs(np.linalg.norm(x) - 1) < 1e-12

        self.Hx = self.effH.apply(x)

    def set_direction (self, d):
        self.d = d
        self.Hd = self.effH.apply(d)

    def move (self, a):
        self.x += a * self.d
        self.Hx += a * self.Hd

        if self.normalize:
            norm = np.linalg.norm(self.x)
            self.x /= norm
            self.Hx /= norm

        self.d = self.Hd = None

    def env (self, a):
        return self.Hx + a * self.Hd

    def val_slope (self, a):
        x = self.x + a * self.d
        env = self.env(a)
        val = inner(env, x).real

        df = 2 * env
        d = self.d

        if self.normalize:
            norm = np.linalg.norm(x)
            val = val / norm**2
            d = d / norm**2

            x = x / norm
            xd = inner (x, d)               # complex number
            d = d - xd * x                  # complex vector
        g = inner(df, d).real             # real number

        return val, g

class cost_function_GP:
    def __init__ (self, L, M, R, L4, x, R4, g, normalize=True):
        self.g = g
        self.func2 = cost_function_xHx (L, M, R, x, normalize)
        self.func4 = cost_function_phi4 (L4, R4, x, normalize)
        self.env0 = self.func2.Hx + g * self.func4.env0

    def set_direction (self, d):
        self.func2.set_direction(d)
        self.func4.set_direction(d)

    def move (self, a):
        self.func2.move(a)
        self.func4.move(a)

    def val_slope (self, a):
        val2, g2 = self.func2.val_slope(a)
        val4, g4 = self.func4.val_slope(a)
        return 2*val2+self.g*val4, 2*g2+self.g*g4

