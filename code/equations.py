from pysph.sph.equation import Equation
from pysph.solver.tools import Tool
from pysph.sph.integrator import IntegratorStep
from pysph.tools.interpolator import InterpolateFunction
from compyle.api import declare
from math import sin, cos, exp, pi
import numpy as np

# Constants
SIGMA = 5.67e-8
SOLAR_CONST = 1366.1 #W/m^2

class DiffusionStep(IntegratorStep):
    """ Stepping"""
    def stage1(self, d_idx, d_aT, d_T, dt):
        d_T[d_idx] += d_aT[d_idx]*dt


class JeongHeatFlux(Equation):
    def initialize(self, d_qx, d_qy, d_qz, d_idx):
        d_qx[d_idx] = 0.0
        d_qy[d_idx] = 0.0
        d_qz[d_idx] = 0.0

    def loop(self, d_qx, d_qy, d_qz, d_T, s_T, d_idx, s_idx, DWIJ,
             s_m, s_rho, RIJ, XIJ):
        Vj = s_m[s_idx] / s_rho[s_idx]
        dT = d_T[d_idx] - s_T[s_idx]

        d_qx[d_idx] += -dT * DWIJ[0] * Vj
        d_qy[d_idx] += -dT * DWIJ[1] * Vj
        d_qz[d_idx] += -dT * DWIJ[2] * Vj


class JeongTempGrad(Equation):
    def initialize(self, d_aT, d_idx):
        d_aT[d_idx] = 0.0

    def loop(self, d_aT, d_idx, s_idx, DWIJ, s_m, s_rho, RIJ, XIJ,
             d_qx, d_qy, d_qz, s_qx, s_qy, s_qz):
        Vj = s_m[s_idx] / s_rho[s_idx]
        qxavg = d_qx[d_idx] + s_qx[s_idx]
        qyavg = d_qy[d_idx] + s_qy[s_idx]
        qzavg = d_qz[d_idx] + s_qz[s_idx]

        dTx2 = DWIJ[0] * qxavg
        dTy2 = DWIJ[1] * qyavg
        dTz2 = DWIJ[2] * qzavg

        d_aT[d_idx] += (dTx2 + dTy2 + dTz2) * Vj


class BrookShawHeatDiffusion(Equation):
    def initialize(self, d_aT, d_idx):
        d_aT[d_idx] = 0.0

    def loop(self, d_T, d_aT, d_idx, s_T, s_idx, DWIJ, s_m, s_rho, RIJ, XIJ):
        Vj = s_m[s_idx] / s_rho[s_idx]
        dT = d_T[d_idx] - s_T[s_idx]

        if RIJ > 1e-14:
            d_aT[d_idx] += 2. * Vj * dT * DWIJ[0] / XIJ[0]


class ClearyHeatDiffusion(Equation):
    def __init__(self, dest, sources, h):
        self.h = h
        self.counter = 0

        super(ClearyHeatDiffusion, self).__init__(dest, sources)

    def initialize(self, d_aT, d_idx):
        d_aT[d_idx] = 0.0

    def loop(self, d_T, d_aT, d_rho, d_cp, d_k, d_idx, s_T, s_idx, DWIJ,
             s_m, s_rho, s_k, R2IJ, XIJ, HIJ):
        xijdotdwij = DWIJ[0]*XIJ[0] + DWIJ[1]*XIJ[1] + DWIJ[2]*XIJ[2]
        tmp = 1./(R2IJ + 1e-4*self.h**2)
        inv_cp = 1./d_cp[0]
        inv_rho = 1./(d_rho[d_idx] * s_rho[s_idx])
        k_avg = 4. * d_k[0] * s_k[0] / (d_k[0] + s_k[0])
        #print(HIJ)
        d_aT[d_idx] += s_m[s_idx] * tmp * inv_cp * inv_rho * k_avg *\
            (d_T[d_idx]-s_T[s_idx]) * xijdotdwij
    #     self.counter += 1
    # def post_loop(self):
    #     print(self.counter)

class UpdateBoundaryFlux(Equation):
    def __init__(self, dest, sources, h0):
        self.h0 = h0
        super(UpdateBoundaryFlux, self).__init__(dest, sources)

    def initialize(self, d_idx, d_qx):
        d_qx[d_idx] = 0.0

    def loop(self, d_qx, d_idx, d_T, d_layer_num, d_k):
        if d_layer_num[d_idx] == 1:
            d_qx[d_idx] = -self.h0 * d_T[d_idx]

class UpdateBoundaryFlux3D(Equation):
    def __init__(self, dest, sources):
        self.SIGMA = SIGMA
        self.flag = 1

        super(UpdateBoundaryFlux3D, self).__init__(dest, sources)

    def initialize(self, d_idx, d_qx):
        d_qx[d_idx] = 0.0

    def loop(self, d_qx, d_idx, d_T, d_layer_num, d_epsilon, d_alpha, d_Fsun):
        if d_layer_num[d_idx] == 1:
            if self.flag:
                #print('update flux is run')
                self.flag=0
            d_qx[d_idx] = -d_epsilon[d_idx] * self.SIGMA * (d_T[d_idx]**4)\
                + d_alpha[d_idx]* d_Fsun[d_idx]

class InterpolateTemperature(Equation):
    def initialize(self, d_idx, d_T, d_number_density):
        d_T[d_idx] = 0.0
        d_number_density[d_idx] = 0.0

    def loop(self, s_idx, d_idx, s_T, d_T, d_layer_num, d_number_density, WIJ, d_tag):
        if d_layer_num[d_idx] == 1:
            d_number_density[d_idx] += WIJ
            d_T[d_idx] += WIJ*s_T[s_idx]
        # if d_T[d_idx] < s_T[s_idx]:
        #     d_T[d_idx] = s_T[s_idx]

    def post_loop(self, d_idx, d_T, d_temp_T, d_number_density):
        if d_number_density[d_idx] > 1e-12:
            #pass
            d_T[d_idx] /= d_number_density[d_idx]
        d_temp_T[d_idx] = d_T[d_idx]

class SetTempGradient(Equation):
    def __init__(self, dest, sources, dx):
        self.dx = dx

        super(SetTempGradient, self).__init__(dest, sources)

    def initialize(self, d_T, d_temp_T, d_idx, d_b_idx, d_qx):
        p_idx = declare('int')
        p_idx = d_b_idx[d_idx]
        d_temp_T[d_idx] = d_temp_T[p_idx]
        d_qx[d_idx] = d_qx[p_idx]
        d_T[d_idx] = 0.0
        #print(str(d_idx) + str(p_idx))

    def loop(self, d_T, d_cp, d_k, d_layer_num, d_idx, d_temp_T, d_qx):
        d_T[d_idx] = d_temp_T[d_idx] + (d_layer_num[d_idx]) * d_qx[d_idx] * self.dx /d_k[0]


# class SetTempGradient(Equation):
#     def __init__(self, dest, sources):
#         self.h = h

#         super(ClearyHeatDiffusion, self).__init__(dest, sources)

#     def initialize(self, d_T, d_idx):
#         d_T[d_idx] = 0.0

#     def loop(self, d_T, d_aT, d_rho, d_cp, d_k, d_idx, s_T, s_idx, DWIJ,
#              s_m, s_rho, s_k, R2IJ, XIJ):
#             temp[0, :] = temp[1, :] + q_l * dx /self.k
#             temp[-1, :] = temp[-2, :] + q_r * dx /self.k
#             temp[:, 0] = temp[:, 1] + q_b * dx /self.k
#             temp[:, -1] = temp[:, -2] + q_t * dx /self.k

class ClearyHeatDiffusionFlux(Equation):
    def __init__(self, dest, sources, h, dx):
        self.h = h
        self.dx = dx

        super(ClearyHeatDiffusionFlux, self).__init__(dest, sources)

    def initialize(self, d_aT, d_idx):
        d_aT[d_idx] = 0.0

    def loop(self, d_T, d_aT, d_rho, d_cp, d_k, d_idx, s_T, s_idx, DWIJ,
             s_m, s_rho, s_k, R2IJ, XIJ, d_qx, RIJ, d_x, d_m_mat):
        #print(d_aT[d_idx], d_idx, s_idx, d_x[d_idx])
        xijdotdwij = DWIJ[0]*XIJ[0] + DWIJ[1]*XIJ[1] + DWIJ[2]*XIJ[2]
        tmp = 1./(R2IJ + 1e-4*self.h**2)
        inv_cp = 1./d_cp[0]
        inv_rho = 1./(d_rho[d_idx] * s_rho[s_idx])
        k_avg = 4. * d_k[0] * s_k[0] / (d_k[0] + s_k[0])

        d_aT[d_idx] += s_m[s_idx] * tmp * inv_cp * inv_rho * k_avg *\
             (d_T[d_idx]-s_T[s_idx]) * xijdotdwij

    def post_loop(self, d_aT,d_cp,d_idx,d_qx, d_rho):
        inv_cp = 1./d_cp[0]
        d_aT[d_idx] +=  d_qx[d_idx] * inv_cp / (d_rho[d_idx] * self.dx) #* XIJ[0]/(RIJ+1e-14)

class ClearyFluxAddition(Equation):
    def __init__(self, dest, sources, h):
        self.h = h

        super(ClearyFluxAddition, self).__init__(dest, sources)

    def initialize(self, d_aT, d_idx):
        d_aT[d_idx] = 0.0

    def loop(self, d_aT,d_cp,d_idx,d_qx):
        inv_cp = 1./d_cp[0]
        d_aT[d_idx] += d_qx[d_idx] * inv_cp

class MonaghanSourceNormFactCalc(Equation):
    def __init__(self, dest, sources, h):
        self.h = h

        super(MonaghanSourceNormFactCalc, self).__init__(dest, sources)

    def initialize(self, d_zeta_inv, d_idx):
        d_zeta_inv[d_idx] = 0.0

    def loop(self, d_zeta_inv,d_idx, s_m, s_rho, WIJ, s_idx):
        d_zeta_inv[d_idx] += s_m[s_idx] * WIJ / s_rho[s_idx]

class MonaghanHeatDiffusionFlux(Equation):
    def __init__(self, dest, sources, h, dx):
        self.h = h
        self.dx = dx

        super(MonaghanHeatDiffusionFlux, self).__init__(dest, sources)

    def initialize(self, d_aT, d_idx):
        d_aT[d_idx] = 0.0

    def loop(self, d_T, d_aT, d_rho, d_cp, d_k, d_idx, s_T, s_idx, DWIJ,
             s_m, s_rho, s_k, R2IJ, XIJ, WIJ, s_qx, RIJ, d_x, d_m_mat, s_zeta_inv):
        #print(d_aT[d_idx], d_idx, s_idx, d_x[d_idx])
        xijdotdwij = DWIJ[0]*XIJ[0] + DWIJ[1]*XIJ[1] + DWIJ[2]*XIJ[2]
        tmp = 1./(R2IJ + 1e-4*self.h**2)
        inv_cp = 1./d_cp[0]
        inv_rho = 1./(d_rho[d_idx] * s_rho[s_idx])
        k_avg = 4. * d_k[0] * s_k[0] / (d_k[0] + s_k[0])

        d_aT[d_idx] += s_m[s_idx] * tmp * inv_cp * inv_rho * k_avg *\
             (d_T[d_idx]-s_T[s_idx]) * xijdotdwij
        d_aT[d_idx] += 2*(inv_cp * s_qx[s_idx] * WIJ ) / (d_rho[d_idx] * s_zeta_inv[s_idx] * self.dx)
        #d_aT[d_idx] += (inv_cp * s_qx[s_idx] * WIJ ) / (d_rho[d_idx] * self.dx)

class CubeFluxUpdate(Tool):
    """A simple tool to periodically remesh a given array of particles onto an
    initial set of points.
    """
    def __init__(self, app, freq=10):
        """
        Parameters
        ----------

        app : pysph.solver.application.Application.
            The application instance.
        freq : int
            Frequency of reinitialization.
        satellite : object

        """
        from pysph.solver.utils import get_array_by_name
        self.freq = freq
        self.satellite = app.satellite
        self.particles = app.particles
        self.boundary_l = get_array_by_name(self.particles, 'boundary_l')
        self.boundary_r = get_array_by_name(self.particles, 'boundary_r')
        self.boundary_t = get_array_by_name(self.particles, 'boundary_t')
        self.boundary_b = get_array_by_name(self.particles, 'boundary_b')
        self.boundary_ft= get_array_by_name(self.particles, 'boundary_ft')
        self.boundary_bk= get_array_by_name(self.particles, 'boundary_bk')
        self.SOLAR_CONST = SOLAR_CONST
        
        # initial run
        self._flux_update(t_i=0)

    def _flux_update(self, t_i):
        #print("tool is run")
        orient_vec, rad_vec = self.satellite.get_orientation_flux_vector(t=t_i)
        Flux = self.SOLAR_CONST * rad_vec

        Area = np.zeros((6,3))
        Area[0,:] = orient_vec[0,:]
        Area[1,:] = -orient_vec[0,:]
        Area[2,:] = orient_vec[1,:]
        Area[3,:] = -orient_vec[1,:]
        Area[4,:] = orient_vec[2,:]
        Area[5,:] = -orient_vec[2,:]
        # Area[1,0] = -orient_vec[0,0]
        # Area[3,1] = -orient_vec[1,1]
        # Area[5,2] = -orient_vec[2,2]

        Fsun = np.zeros(6)
        for i in range(6):
            AdotF = np.dot(Area[i], Flux)
            if AdotF < 0:
                Fsun[i] = abs(AdotF)
        # print('Area', Area)
        # print('rad vec', rad_vec)
        # print(Fsun)

        self.boundary_l.Fsun[:] = Fsun[0]
        self.boundary_r.Fsun[:] = Fsun[1]
        self.boundary_t.Fsun[:] = Fsun[2]
        self.boundary_b.Fsun[:] = Fsun[3]
        self.boundary_ft.Fsun[:] = Fsun[4]
        self.boundary_bk.Fsun[:] = Fsun[5]


    def pre_step(self, solver):
        #print(solver.count)
        if solver.count % self.freq == 0 and solver.count > 0:
            t_i = solver.t
            self._flux_update(t_i=t_i)
            