""" Cube Sat Thermal Application"""
import numpy as np
import os

from pysph.base.kernels import QuinticSpline
from pysph.solver.application import Application
from pysph.solver.solver import Solver

from pysph.sph.equation import Group
from pysph.sph.basic_equations import SummationDensity
from pysph.sph.wc.kernel_correction import GradientCorrection, GradientCorrectionPreStep
from pysph.sph.integrator import EulerIntegrator
from pysph.base.utils import get_particle_array_wcsph, get_particle_array
from pysph.sph.scheme import add_bool_argument
from pysph.tools.geometry import remove_overlap_particles

from satellite_flux import SatelliteFlux

from equations import *
import matplotlib
# matplotlib.use('pdf')



class CubeSatThermal(Application):
    """ Heat Diffusion Application"""
    def initialize(self):
        self.dx = 0.025
        self.rho = 2770.#2.7e3
        self.h = 1.0 * self.dx
        self.dim = 3
        self.nl = 3
        self.method = 1
        self.hdx = 1.2
        self.ts = 0.15
        self.L = 0.632
        self.B = 0.632
        self.H = 0.632
        self.thickness = 0.01 # = 10mm
        self.t0 = 0.    # Ambient Temparature
        self.t1 = 215.0    # Temperature of box
        self.cp = 961.2#900.  # J/kg.K
        self.k = 167.9#205.   # J/m.K
        self.absorptivity = 0.42
        self.emissivity = 0.87
        self.satellite = SatelliteFlux(epoch=2433282) # Create a satellite model (default)

    def add_user_options(self, group):
        group.add_argument(
            "--n", action="store", type=int, dest="n", default=100,
            help="Number of particles."
        )
        group.add_argument(
            "--hdx", action="store", type=float, dest="hdx", default=1.0,
            help="h/dx."
        )
        group.add_argument(
            "--dim", action="store", type=int, dest="dim", default=3,
            help="Dimension of problem."
        )
        group.add_argument(
            "--ts", action="store", type=float, dest="ts", default=0.15,
            help="Time step factor."
        )
        group.add_argument(
            '--grad-correction', action="store_true", dest='grad_correction',
            default=False, help="Use kernel gradient correction.",
        )
        group.add_argument(
            '--sum-density', action="store_true", dest='sum_density',
            default=False, help="Use summation density.",
        )

    def consume_user_options(self):
        self.dx = self.L/float(self.options.n)
        self.hdx = self.options.hdx
        self.h = self.hdx * self.dx
        # self.h = 0.268 * self.dx**(0.5) # schwaiger
        self.dim = self.options.dim
        self.ts = self.options.ts

    def _create_3d_surf(self):
        dx = self.dx
        x, y, z = np.mgrid[dx/2:self.L:dx, dx/2:self.B:dx, dx/2:self.H:dx]

        tk = self.thickness
        if self.thickness > 0:
            box = ~((x >= tk) & (x <= self.L - tk) & (y >= tk) & (y <= self.L - tk) & (z >= tk) & (z <= self.L - tk))
            xs, ys, zs = x[box], y[box], z[box]
        else:
            xs, ys, zs = x[:], y[:], z[:]
        rho = self.rho
        h = self.h
        m = rho * self.dx**self.dim
        w_dx = self.nl * dx

        slab = get_particle_array(
            name='slab', x=xs, y=ys, z=zs, m=m, h=h, rho=rho)

        # left
        xb, yb, zb = x[0,:,:] - dx, y[0,:, :], z[0,:,:]
        boundary_l = get_particle_array_wcsph(
            name='boundary_l', x=xb, y=yb, z=zb, m=m, h=h, rho=rho)
        # right
        xb, yb, zb = x[-1,:, :] + dx, y[-1,:,:], z[-1,:,:]
        boundary_r = get_particle_array_wcsph(
            name='boundary_r', x=xb, y=yb, z=zb, m=m, h=h, rho=rho)        
        # top
        xb, yb, zb = x[:,-1,:] , y[:,-1,:] + dx, z[:,-1,:]
        boundary_t = get_particle_array_wcsph(
            name='boundary_t', x=xb, y=yb, z=zb, m=m, h=h, rho=rho) 
        # bottom
        xb, yb, zb = x[:,0,:] , y[:,0,:] - dx, z[:,0,:]
        boundary_b = get_particle_array_wcsph(
            name='boundary_b', x=xb, y=yb, z=zb, m=m, h=h, rho=rho)
        # front
        xb, yb, zb = x[:,:,-1] , y[:,:,-1], z[:,:,-1] + dx
        boundary_ft = get_particle_array_wcsph(
            name='boundary_ft', x=xb, y=yb, z=zb, m=m, h=h, rho=rho) 
        # back
        xb, yb, zb = x[:,:,0] , y[:,:,0], z[:,:,0] - dx
        boundary_bk = get_particle_array_wcsph(
            name='boundary_bk', x=xb, y=yb, z=zb, m=m, h=h, rho=rho)

        addl_props = ['layer_num', 'b_idx']
        for prop in addl_props:
            boundary_l.add_property(prop, type='int')
            boundary_r.add_property(prop, type='int')
            boundary_t.add_property(prop, type='int')
            boundary_b.add_property(prop, type='int')
            boundary_ft.add_property(prop, type='int')
            boundary_bk.add_property(prop, type='int')
        boundary_l.layer_num[:] = 1
        boundary_r.layer_num[:] = 1
        boundary_t.layer_num[:] = 1
        boundary_b.layer_num[:] = 1
        boundary_ft.layer_num[:] = 1
        boundary_bk.layer_num[:] = 1

        boundary_l.b_idx[:] = parent_idx_l = range(boundary_l.get_number_of_particles())
        boundary_r.b_idx[:] = parent_idx_r = range(boundary_r.get_number_of_particles())
        boundary_t.b_idx[:] = parent_idx_t = range(boundary_t.get_number_of_particles())
        boundary_b.b_idx[:] = parent_idx_b = range(boundary_b.get_number_of_particles())
        boundary_ft.b_idx[:]= parent_idx_ft = range(boundary_ft.get_number_of_particles())
        boundary_bk.b_idx[:]= parent_idx_bk = range(boundary_bk.get_number_of_particles())

        for i in range(self.nl - 1):
            n = i + 2
            # left
            xb, yb, zb = x[0,:,:] - n * dx, y[0,:,:], z[0,:,:]
            bl = get_particle_array(x=xb, y=yb, z=zb, layer_num=n, b_idx=parent_idx_l,
                m=m, h=h, rho=rho)
            boundary_l.append_parray(bl)
            # right
            xb, yb, zb = x[-1,:,:] + n * dx, y[-1,:,:], z[-1,:,:]
            br = get_particle_array(x=xb, y=yb, z=zb, layer_num=n, b_idx=parent_idx_r,
                m=m, h=h, rho=rho)
            boundary_r.append_parray(br)
            # top
            xb, yb, zb = x[:,-1,:] , y[:,-1,:] + n * dx, z[:,-1,:]
            bt = get_particle_array(x=xb, y=yb, z=zb, layer_num=n, b_idx=parent_idx_t,
                m=m, h=h, rho=rho)
            boundary_t.append_parray(bt)
            # bottom
            xb, yb, zb = x[:,0,:] , y[:,0,:] - n * dx, z[:,0,:]
            bb = get_particle_array(x=xb, y=yb, z=zb, layer_num=n, b_idx=parent_idx_b,
                m=m, h=h, rho=rho)
            boundary_b.append_parray(bb)
            # front
            xb, yb, zb = x[:,:,-1] , y[:,:,-1], z[:,:,-1] + n*dx
            bft = get_particle_array(x=xb, y=yb, z=zb, layer_num=n, b_idx=parent_idx_ft,
                m=m, h=h, rho=rho)
            boundary_ft.append_parray(bft)
            # back
            xb, yb, zb = x[:,:,0] , y[:,:,0], z[:,:,0] - n*dx
            bbk = get_particle_array(x=xb, y=yb, z=zb, layer_num=n, b_idx=parent_idx_bk,
                m=m, h=h, rho=rho)
            boundary_bk.append_parray(bbk)

        props = ['T', 'aT','qx', 'temp_T', 'number_density', 'Fsun', 'alpha', 'epsilon']
        for prop in props:
            slab.add_property(prop)
            boundary_l.add_property(prop)
            boundary_r.add_property(prop)
            boundary_t.add_property(prop)
            boundary_b.add_property(prop)
            boundary_ft.add_property(prop)
            boundary_bk.add_property(prop)

        # properties for kgc
        slab.add_property('m_mat', stride=9)
        boundary_l.add_property('m_mat', stride=9)
        boundary_r.add_property('m_mat', stride=9)
        boundary_t.add_property('m_mat', stride=9)
        boundary_b.add_property('m_mat', stride=9)
        boundary_ft.add_property('m_mat', stride=9)
        boundary_bk.add_property('m_mat', stride=9)

        slab.add_output_arrays(props)
        boundary_l.add_output_arrays(props + addl_props)
        boundary_r.add_output_arrays(props + addl_props)
        boundary_t.add_output_arrays(props + addl_props)
        boundary_b.add_output_arrays(props + addl_props)
        boundary_ft.add_output_arrays(props + addl_props)
        boundary_bk.add_output_arrays(props + addl_props)

        
        for cnst, cnst_val in zip(['cp', 'k'], [self.cp, self.k]):
            slab.add_constant(cnst, cnst_val)
            boundary_l.add_constant(cnst, cnst_val)
            boundary_r.add_constant(cnst, cnst_val)
            boundary_t.add_constant(cnst, cnst_val)
            boundary_b.add_constant(cnst, cnst_val)
            boundary_ft.add_constant(cnst, cnst_val)
            boundary_bk.add_constant(cnst, cnst_val)

        alpha = self.absorptivity
        epsilon = self.emissivity
        boundary_l.alpha[:] = alpha
        boundary_r.alpha[:] = alpha
        boundary_t.alpha[:] = alpha
        boundary_b.alpha[:] = alpha
        boundary_ft.alpha[:]= alpha 
        boundary_bk.alpha[:]= alpha 
        boundary_l.epsilon[:] = epsilon
        boundary_r.epsilon[:] = epsilon
        boundary_t.epsilon[:] = epsilon
        boundary_b.epsilon[:] = epsilon
        boundary_ft.epsilon[:]= epsilon
        boundary_bk.epsilon[:]= epsilon

        slab.T[:] = self.t1
        boundary_l.T[:] = boundary_l.temp_T[:] = self.t1
        boundary_r.T[:] = boundary_r.temp_T[:] = self.t1
        boundary_t.T[:] = boundary_t.temp_T[:] = self.t1
        boundary_b.T[:] = boundary_b.temp_T[:] = self.t1
        boundary_ft.T[:] = boundary_ft.temp_T[:] = self.t1
        boundary_bk.T[:] = boundary_bk.temp_T[:] = self.t1

        return [slab, boundary_l, boundary_r, boundary_t, boundary_b, boundary_ft, boundary_bk]

    def create_particles(self):
        if self.dim == 3:
            pa = self._create_3d_surf()

        pa = self.update_initial_properties(pa)
        return pa

    def create_solver(self):
        kernel = QuinticSpline(dim=self.dim)
        alpha = self.k / (self.rho * self.cp)
        ts = self.ts
        dt = self.h**2 * ts/alpha
        tf = 3.0
        if self.dim == 3 and self.method == 1:
            integrator = EulerIntegrator(slab=DiffusionStep())
        solver = Solver(
            kernel=kernel, dim=self.dim, integrator=integrator,
            dt=dt, tf=tf
        )
        return solver

    def cleary(self):
        equations = []
        if self.dim == 3:
            all_entities = ['slab', 'boundary_l', 'boundary_r', 'boundary_t', 'boundary_b',
                            'boundary_ft', 'boundary_bk']
            equations = [
                Group(
                    equations=[
                                InterpolateTemperature(dest='boundary_l', sources=['slab']),
                                InterpolateTemperature(dest='boundary_r', sources=['slab']),
                                InterpolateTemperature(dest='boundary_t', sources=['slab']),
                                InterpolateTemperature(dest='boundary_b', sources=['slab']),
                                InterpolateTemperature(dest='boundary_ft', sources=['slab']),
                                InterpolateTemperature(dest='boundary_bk', sources=['slab'])
                    ],
                real=False),
                Group(
                    equations=[
                                UpdateBoundaryFlux3D(dest='boundary_l', sources=['slab']),
                                UpdateBoundaryFlux3D(dest='boundary_r', sources=['slab']),
                                UpdateBoundaryFlux3D(dest='boundary_t', sources=['slab']),
                                UpdateBoundaryFlux3D(dest='boundary_b', sources=['slab']),
                                UpdateBoundaryFlux3D(dest='boundary_ft', sources=['slab']),
                                UpdateBoundaryFlux3D(dest='boundary_bk', sources=['slab'])
                    ],
                real=False),                                            
                Group(
                    equations=[
                                SetTempGradient(dest='boundary_l', sources=None, dx=self.dx),
                                SetTempGradient(dest='boundary_r', sources=None, dx=self.dx),
                                SetTempGradient(dest='boundary_t', sources=None, dx=self.dx),
                                SetTempGradient(dest='boundary_b', sources=None, dx=self.dx),
                                SetTempGradient(dest='boundary_ft', sources=None, dx=self.dx),
                                SetTempGradient(dest='boundary_bk', sources=None, dx=self.dx)
                    ],
                real=False),
                Group(
                    equations=[
                                ClearyHeatDiffusion(dest='slab', sources=all_entities,
                                             h=self.h)
                    ],
                real=False)
            ]
            # if gradient correction is enabled
            if self.options.grad_correction:
                print("adding gradient correction")
                eqn1 = Group(equations=[
                    GradientCorrectionPreStep(dest='slab', sources=all_entities, dim=self.dim)
                    ],real=False)
                eqn2 = GradientCorrection(dest='slab', sources=all_entities, dim=self.dim, tol=5.0)
                equations.insert(0, eqn1)
                equations[-1].equations.insert(0, eqn2)

        return equations

    def update_initial_properties(self, pa):
        from pysph.tools.sph_evaluator import SPHEvaluator
        kernel = QuinticSpline(dim=self.dim)
        #name = pa.name
        if self.options.sum_density:
            if self.dim == 3:
                all_entities = ['slab', 'boundary_l', 'boundary_r', 'boundary_t', 'boundary_b',
                                'boundary_ft', 'boundary_bk']
                sph_eval = SPHEvaluator(
                    arrays= pa, equations=[
                        Group(equations=[
                                SummationDensity(dest='slab', sources=all_entities),
                                SummationDensity(dest='boundary_l', sources=['boundary_l','slab']),
                                SummationDensity(dest='boundary_r', sources=['boundary_r','slab']),
                                SummationDensity(dest='boundary_t', sources=['boundary_t','slab']),
                                SummationDensity(dest='boundary_b', sources=['boundary_b','slab']),
                                SummationDensity(dest='boundary_ft', sources=['boundary_ft','slab']),
                                SummationDensity(dest='boundary_bk', sources=['boundary_bk','slab'])
                        ],real=False)],
                    dim=self.dim, kernel=kernel
                )
                sph_eval.evaluate()
        return pa

    def create_equations(self):
        eqns = self.cleary()
        return eqns

    def create_tools(self):
        flux_update_tool = CubeFluxUpdate(
            self, freq=100
        )
        return [flux_update_tool]

    def post_process(self, info_fname):
        self.read_info(info_fname)
        if len(self.output_files) == 0:
            return
        self.res = os.path.join(self.output_dir, 'results.npz')
        x, comp, exact = self._l2_norm()
        np.savez(self.res, x=x, comp=comp, exact=exact)
        # self.plot_exact_vs_computed()

    def plot_exact_vs_computed(self):
        from pysph.solver.utils import load, iter_output
        from matplotlib import pyplot as plt

        plt.figure()
        for sd, arrays in iter_output(self.output_files):

            slab = arrays['slab']
            boundary = arrays['boundary']
            tf = sd['t']
            xs = slab.x[:]
            ys = slab.y[:]
            comps = slab.T[:]

            xb = boundary.x[:]
            yb = boundary.y[:]
            compb = boundary.T[:]

            x = np.concatenate((xs, xb))
            y = np.concatenate((ys, yb))
            comp = np.concatenate((comps, compb))
            ids = np.argsort(x)
            x = x[ids]
            y = y[ids]
            comp = comp[ids]
            T = np.sin(x+y) * np.cos(tf)
            plt.cla()
            plt.plot(x, T, label='exact')
            plt.plot(x, comp, label='comp')
            plt.legend()
            plt.grid()
            plt.pause(0.2)

    def _l2_norm(self):
        from pysph.solver.utils import load
        from matplotlib import pyplot as plt
        import numpy as np

        #plt.figure()
        #for i in range(len(self.output_files)):
        data = load(self.output_files[-1])
        slab = data['arrays']['slab']
        tf = data['solver_data']['t']
        x = np.copy(slab.x)
        y = np.copy(slab.x)
        z = np.copy(slab.x)
        comp = np.copy(slab.T)
        exact = np.zeros_like(comp)
        if self.dim == 3:
            t, err = [], []
            Tmin, Tmax, Tavg = [],[],[]
            T1max, T2max, T3max, T4max, T5max, T6max = [],[],[],[],[],[]
            T1min, T2min, T3min, T4min, T5min, T6min = [],[],[],[],[],[]
            for i in range(0,len(self.output_files)):
                data = load(self.output_files[i])
                slab = data['arrays']['slab']
                boundary_l = data['arrays']['boundary_l'].temp_T
                boundary_r = data['arrays']['boundary_r'].temp_T
                boundary_t = data['arrays']['boundary_t'].temp_T
                boundary_b = data['arrays']['boundary_b'].temp_T
                boundary_ft = data['arrays']['boundary_ft'].temp_T
                boundary_bk = data['arrays']['boundary_bk'].temp_T
                tf = data['solver_data']['t']
                comp_i = np.copy(slab.T)
                #rho_p  = np.copy(slab.rho)
                # F_sun_f = np.array([boundary_l.Fsun[0], boundary_r.Fsun[0],
                #             boundary_t.Fsun[0], boundary_b.Fsun[0],
                #             boundary_ft.Fsun[0], boundary_bk.Fsun[0]                
                #             ])
                F_sun_f = np.array([1367])
                #print(F_sun_f)
                #print(slab.aT[:])
                #print(comp)
                TE = np.sum((comp_i - self.t1) * self.cp * self.rho * self.dx**self.dim)
                Ein = np.sum(F_sun_f * tf * self.L*self.B * self.absorptivity)
                er = abs(Ein-TE)/(Ein + 1e-12)
                # if er > 10:
                #     break
                t.append(tf)
                err.append(er)
                Tma, Tmi = np.max(comp_i)-273.15, np.min(comp_i)-273.15
                Tav = (np.sum(comp_i)/len(comp_i))-273.15
                Tmax.append(Tma)
                Tmin.append(Tmi)
                Tavg.append(Tav)

                T1max.append(np.max(boundary_l )-273.15)
                T2max.append(np.max(boundary_r )-273.15)
                T3max.append(np.max(boundary_t )-273.15)
                T4max.append(np.max(boundary_b )-273.15)
                T5max.append(np.max(boundary_ft)-273.15)
                T6max.append(np.max(boundary_bk)-273.15)
                T1min.append(np.min(boundary_l )-273.15)
                T2min.append(np.min(boundary_r )-273.15)
                T3min.append(np.min(boundary_t )-273.15)
                T4min.append(np.min(boundary_b )-273.15)
                T5min.append(np.min(boundary_ft)-273.15)
                T6min.append(np.min(boundary_bk)-273.15)

                # print(f'At t = {tf}:')
                # print(f'Ein = {Ein}, TE = {TE}, Abs. E = {er}')
                if i == len(self.output_files) - 1:
                    print(f'Tmax={Tma}, Tmin={Tmi}, Tavg={Tav}')
                # print(' ')
                    print('mass', np.sum(self.rho*np.ones_like(comp_i) * self.dx**self.dim))
                    print('volume',np.sum(np.ones_like(comp_i) * self.dx**self.dim))
                    #print('m v (act)', self.rho*self.L**3, self.L**3)
            plt.figure()
            plt.plot(t, err, label='absolute error')
            plt.plot(t, err, '.')
            plt.xlabel('t')
            plt.ylabel('abs error')
            plt.legend()
            plt.grid()
            fig = os.path.join(self.output_dir, "comp.png")
            plt.savefig(fig, dpi=300)
            plt.close()

            #plot temperature
            plt.figure()
            plt.plot(t, Tmax, '.', label=r'$T_{max}$')
            #plt.plot(t, Tmax, '.')
            plt.plot(t, Tmin, '.', label=r'$T_{min}$')
            #plt.plot(t, Tmin, '.')
            #plt.plot(t, Tavg, label=r'$T_{avg}$')
            #plt.plot(t, Tavg, '.')
            # file_data = np.loadtxt(fname='code/temp_500sec.txt', skiprows=1)
            # T_a, Tmin_a, Tmax_a, Tavg_a = file_data[:,1], file_data[:,2], file_data[:,3], file_data[:,4]
            # plt.plot(T_a, Tmin_a, '--')
            # #plt.plot(t, Tmax, '.')
            # plt.plot(T_a, Tmax_a, '--')
            # #plt.plot(t, Tmin, '.')
            # plt.plot(T_a, Tavg_a, '--')
            plt.plot(t, T1max)
            plt.plot(t, T2max)
            plt.plot(t, T3max)
            plt.plot(t, T4max)
            plt.plot(t, T5max)
            plt.plot(t, T6max)
            plt.plot(t, T1min)
            plt.plot(t, T2min)
            plt.plot(t, T3min)
            plt.plot(t, T4min)
            plt.plot(t, T5min)
            plt.plot(t, T6min)
            
            plt.xlabel('t')
            plt.ylabel(r'Temperature (°C)')
            plt.legend()
            plt.grid()
            fig = os.path.join(self.output_dir, "temp.png")
            plt.savefig(fig, dpi=300)
            plt.close()

            # plt.figure()
            # mask = ((x > 0.4) & (x < 0.4 + self.dx))
            # mask2 = ((y > 0.4) & (y < 0.4 + self.dx))
            # m = z[mask & mask2]
            # print(m[:])
            # plt.scatter(y[mask], z[mask])
            # fig = os.path.join(self.output_dir, "comp2.png")
            # plt.show()
            # plt.savefig(fig, dpi=300)
            #plt.close()
            
        return x, comp, exact

        # print('not this')
        # x = list(x)
        # comp = list(comp)
        # exact = list(exact)
        # x, comp, exact = list(map(np.asarray, (x, comp, exact)))

        # if self.dim == 3 and self.mms == 3:
        #     _x = np.reshape(x, (self.options.n, self.options.n))
        #     _comp = np.reshape(comp, (self.options.n, self.options.n))
        #     _exact = np.reshape(exact, (self.options.n, self.options.n))
        #     x_diag = np.diag(np.flip(_x))
        #     str_tf = 't='+ str(tf)
        #     plt.plot(x_diag, np.diag(np.flip(_comp)), label='Numerical'+ ' ' + str_tf)
        #     plt.plot(x_diag, np.diag(np.flip(_exact)), label='Exact'+ ' ' + str_tf)
        #     l2 = np.sum((exact - comp)**2)/len(comp)
        #     print('at ' + str_tf + ' : ' + str(l2))

        # plt.xlabel('x')
        # plt.ylabel('T')
        # plt.legend()
        # plt.grid()
        # fig = os.path.join(self.output_dir, "comp.eps")
        # plt.savefig(fig, dpi=300)
        # plt.close()
        # # print('L2 norm: ', l2)
        # # print('L2 norm (Ansys): ', l2_a)
        
        # return x, comp, exact


if __name__ == '__main__':
    app = CubeSatThermal()
    #app.run()
    app.post_process(app.info_filename)
