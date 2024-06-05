import matplotlib.pyplot as plt
from shenfun import *
from ChannelFlow2D import KMM


class OrrSommerfeld(KMM):

    def __init__(self, N=(32, 32), domain=((-1, 1), (0, 2*np.pi)), alpha=1, Re=8000., Rem=0.1, B_0 = (0,0), method='G',
                 dt=0.1, conv=0, modplot=100, modsave=1e8, moderror=100, filename='KMM',
                 family='C', padding_factor=(1, 1.5), checkpoint=1000, timestepper='IMEXRK3'):
        KMM.__init__(self, N=N, domain=domain, nu=1/Re, dt=dt, conv=conv, modplot=modplot,
                     modsave=modsave, moderror=moderror, filename=filename, family=family,
                     padding_factor=padding_factor, checkpoint=checkpoint, timestepper=timestepper,
                     dpdy=-2/Re)
        self.alfa = alpha
        self.Re = Re
        self.Rem = Rem
        self.B_0 = B_0
        self.method = method


        # New spaces and Functions used by MHD
        self.BX = FunctionSpace(N[0], family, bc=(self.B_0[0], self.B_0[0]), domain=self.D0.domain)
        self.BY = FunctionSpace(N[0], family, bc=(self.B_0[1], self.B_0[1]), domain=self.D0.domain)
        self.TBX = TensorProductSpace(comm, (self.BX, self.F1), collapse_fourier=False, slab=True, modify_spaces_inplace=True)
        self.TBY = TensorProductSpace(comm, (self.BY, self.F1), collapse_fourier=False, slab=True, modify_spaces_inplace=True)
        self.VB = VectorSpace([self.TBX, self.TBY])      # B solution
        self.B_ = Function(self.VB)
        self.NB_ = Function(self.CD)      # Convection nabla(B) dot B
        self.NBu_ = Function(self.CD)     # Convection nabla(B) dot u
        self.NuB_ = Function(self.CD)     # Convection nabla(u) dot B
        self.Bb = Array(self.VB)

        # Classes for fast projections used by convection
        self.dB0dx = Project(Dx(self.B_[0], 0, 1), self.TC)
        self.dB0dy = Project(Dx(self.B_[0], 1, 1), self.TD)
        self.dB1dx = Project(Dx(self.B_[1], 0, 1), self.TC)
        self.dB1dy = Project(Dx(self.B_[1], 1, 1), self.TD)
        self.divb = Project(div(self.B_), self.TC)

        self.work_b = CachedArrayDict()

        test_u = self.TB.get_testspace(self.method)           
        v = TestFunction(test_u)

        if self.method == "G":
            sol1 = chebyshev.la.Biharmonic if self.B0.family() == 'chebyshev' else la.SolverGeneric1ND
        elif self.method == "PG":
            sol1 = la.SolverGeneric1ND
        
        self.pdes = {

            'u': self.PDE(v,                                   # test function
                     div(grad(self.u_[0])),               # u
                     lambda f: 1/self.Re*div(grad(f)),      # linear operator on u
                     [Dx(Dx(self.H_[1], 0, 1), 1, 1)-Dx(self.H_[0], 1, 2), Dx(self.NB_[0], 1, 2)-Dx(Dx(self.NB_[1], 0, 1), 1, 1)],
                     dt=self.dt,
                     solver=sol1),

        }

        if comm.Get_rank() == 0:
            test_v0 = self.D00.get_testspace(self.method)
            v0 = TestFunction(test_v0)
            self.h1 = Function(self.D00)  # Copy from H_[1, :, 0, 0] (cannot use view since not contiguous)
            self.b1 = Function(self.D00)  # Copy from NB_[1, :, 0, 0] (cannot use view since not contiguous)
            self.source = Array(self.C00)
            self.source[:] = -self.dpdy        # dpdy set by subclass
            if self.method == "G":
                sol = chebyshev.la.Helmholtz if self.B0.family() == 'chebyshev' else la.Solver
            elif self.method == "PG":
                sol = la.Solver
            
            self.pdes1d = {
                'v0': self.PDE(v0,
                          self.v00,
                          lambda f: 1/self.Re*div(grad(f)),
                          [-Expr(self.h1), source, Expr(self.b1)],
                          dt=self.dt,
                          solver=sol),
            }

        # MHD equations
        test_bx = self.TBX.get_testspace(self.method)
        test_by = self.TBY.get_testspace(self.method)
        tbx = TestFunction(test_bx) 
        tby = TestFunction(test_by)  

        if self.method == "G":
            sol2 = chebyshev.la.Helmholtz if self.B0.family() == 'chebyshev' else la.SolverGeneric1ND
        elif self.method == "PG":
            sol2 = la.SolverGeneric1ND

        self.pdes['B0'] = self.PDE(tbx,
                                   self.B_[0],
                                  lambda f: (1/self.Rem)*div(grad(f)),
                                   [Expr(self.NBu_[0]), -Expr(self.NuB_[0])],
                                   dt=self.dt,
                                   solver=sol2,
                                   )

        self.pdes['B1'] = self.PDE(tby,
                                   self.B_[1],
                                   lambda f: (1/self.Rem)*div(grad(f)),
                                   [Expr(self.NBu_[1]), -Expr(self.NuB_[1])],
                                   dt=self.dt,
                                   solver=sol2,
                                   )



    def convection(self):
        up = self.u_.backward(padding_factor=self.padding_factor)

        dudxp = self.dudx().backward(padding_factor=self.padding_factor)
        dudyp = self.dudy().backward(padding_factor=self.padding_factor)
        dvdxp = self.dvdx().backward(padding_factor=self.padding_factor)
        dvdyp = self.dvdy().backward(padding_factor=self.padding_factor)

        H = self.H_
        
        H[0] = self.TDp.forward(up[0]*dudxp+up[1]*dudyp, H[0])
        H[1] = self.TDp.forward(up[0]*dvdxp+up[1]*dvdyp, H[1])
        
        H.mask_nyquist(self.mask)
        
        BB = self.NB_
        Bu = self.NBu_
        uB = self.NuB_
        Bp = self.B_.backward(padding_factor=self.padding_factor)

        dB0dxp = self.dB0dx().backward(padding_factor=self.padding_factor)
        dB0dyp = self.dB0dy().backward(padding_factor=self.padding_factor)
        dB1dxp = self.dB1dx().backward(padding_factor=self.padding_factor)
        dB1dyp = self.dB1dy().backward(padding_factor=self.padding_factor)

        BB[0] = self.TDp.forward(Bp[0]*dB0dxp+Bp[1]*dB0dyp, BB[0])
        BB[1] = self.TDp.forward(Bp[0]*dB1dxp+Bp[1]*dB1dyp, BB[1])
        BB.mask_nyquist(self.mask)
        uB[0] = self.TDp.forward(up[0]*dB0dxp+up[1]*dB0dyp, uB[0])
        uB[1] = self.TDp.forward(up[0]*dB1dxp+up[1]*dB1dyp, uB[1])
        uB.mask_nyquist(self.mask)
        Bu[0] = self.TDp.forward(Bp[0]*dudxp+Bp[1]*dudyp, Bu[0])
        Bu[1] = self.TDp.forward(Bp[0]*dvdxp+Bp[1]*dvdyp, Bu[1])
        Bu.mask_nyquist(self.mask)
        
    def compute_v(self, rk):
        u = self.u_
        if comm.Get_rank() == 0:
            self.v00[:] = u[1, :, 0].real
            self.h1[:] = self.H_[1, :, 0].real
            self.b1[:] = self.NB_[1, :, 0].real

        # Find velocity components v from div. constraint
        u[1] = 1j*self.dudx()/self.K[1]

        # Still have to compute for wavenumber = 0, 0
        if comm.Get_rank() == 0:
            # v component
            self.pdes1d['v0'].compute_rhs(rk)
            u[1, :, 0] = self.pdes1d['v0'].solve_step(rk)

        return u

    def initialize(self, from_checkpoint=False):
        if from_checkpoint:
            return self.init_from_checkpoint()
        from OrrSommerfeldMHD_eigs import OrrSommerfeldMHD
        self.MOS = MOS = OrrSommerfeldMHD(alfa=self.alfa, Re=self.Re, Rem=self.Rem, By=self.B_0[1], N=128, test=self.method, trial=self.method)
        eigvals, eigvectors = MOS.solve(False)
        MOS.eigvals, MOS.eigvectors = eigvals, eigvectors
        self.initOS(MOS, eigvals, eigvectors, self.ub, self.Bb)
        self.u_ = self.ub.forward(self.u_)
        self.B_ = self.Bb.forward(self.B_)
        self.ub = self.u_.backward(self.ub)
        self.Bb = self.B_.backward(self.Bb)
        # Compute convection from data in context (i.e., context.U_hat and context.g)
        # This is the convection at t=0

        ones_x = np.ones_like(self.X[0])
        self.e0_u = 0.5*dx(self.ub[0]**2+(self.ub[1]-(1-self.X[0]**2))**2)
        self.e0_B = 0.5*dx(self.Bb[0]**2+(self.Bb[1] - self.B_0[1]*ones_x)**2)
        self.acc_u = np.zeros(1)
        self.acc_B = np.zeros(1)
        self.print_energy_and_divergence(0, 0)
        return 0, 0

    def initOS(self, OS, eigvals, eigvectors, U, B, t=0.):
        X = self.X
        x = X[0][:, 0].copy()
        ones_x = np.ones_like(x)
        eigval, phi_u, dphidy_u, phi_b, dphidy_b = OS.interp(x, eigvals, eigvectors, eigval=1, verbose=False)
        OS.eigval = eigval
        for j in range(U.shape[2]):
            y = X[1][0, j]
            v = (1-x**2) + 1e-7*np.real(dphidy_u*np.exp(1j*self.alfa*(y-eigval*t)))
            u = -1e-7*np.real(1j*self.alfa*phi_u*np.exp(1j*self.alfa*(y-eigval*t)))
            U[0, :, j] = u
            U[1, :, j] = v
            bx =  -1e-7*np.real(1j*self.alfa*phi_b*np.exp(1j*self.alfa*(y-eigval*t)))
            by = self.B_0[1]*ones_x + 1e-7*np.real(dphidy_b*np.exp(1j*self.alfa*(y-eigval*t))) 
            B[0, :, j] = bx
            B[1, :, j] = by

    def compute_error(self, t):
        ub = self.u_.backward(self.ub)
        Bb = self.B_.backward(self.Bb)
        ones_x = np.ones_like(self.X[0])
        e1_u = 0.5*dx(ub[0]**2 + (ub[1] - (1-self.X[0]**2))**2)
        e1_B = 0.5*dx(Bb[0]**2) + 0.5*dx((Bb[1] - self.B_0[1]*ones_x)**2)
        #exact = np.exp(2.378465*np.imag(self.MOS.eigval)*t)
        exact = np.exp(2*np.imag(self.MOS.eigval)*t)

        u0 = self.work[(ub, 0, True)]
        b0 = self.work_b[(Bb, 0, True)]
        self.initOS(self.MOS, self.MOS.eigvals, self.MOS.eigvectors, u0, b0, t=t)
        self.u_ = self.ub.forward(self.u_)
        self.B_ = self.Bb.forward(self.B_)
        self.ub = self.u_.backward(self.ub)
        self.Bb = self.B_.backward(self.Bb)
        e2_u = 0.5*dx((ub[0] - u0[0])**2 + (ub[1] - u0[1])**2)
        e2_B = 0.5*dx((Bb[0] - b0[0])**2 + (Bb[1] - b0[1])**2)

        return e1_u, e2_u, exact, e1_B, e2_B

    def init_plots(self):
        '''
        ub = self.u_.backward(self.ub)
        self.im1 = 1
        if comm.Get_rank() == 0 and comm.Get_size() == 1:
            plt.figure(1, figsize=(6, 3))
            self.im1 = plt.contourf(self.X[1], self.X[0], ub[0], 100)
            plt.colorbar(self.im1)
            plt.draw()

            plt.figure(2, figsize=(6, 3))
            self.im2 = plt.contourf(self.X[1], self.X[0], ub[1] - (1-self.X[0]**2), 100)
            plt.colorbar(self.im2)
            plt.draw()

            plt.figure(3, figsize=(6, 3))
            self.im3 = plt.quiver(self.X[1], self.X[0], ub[1]-(1-self.X[0]**2), ub[0])
            plt.colorbar(self.im3)
            plt.draw()
        '''
    def plot(self, t, tstep):
        '''
        if self.im1 is None and self.modplot > 0:
            self.init_plots()

        if tstep % self.modplot == 0 and self.modplot > 0:
            if comm.Get_rank() == 0 and comm.Get_size() == 1:
                ub = self.u_.backward(self.ub)
                X = self.X
                self.im1.axes.clear()
                self.im1.axes.contourf(X[1], X[0], ub[0], 100)
                self.im1.autoscale()
                self.im2.axes.clear()
                self.im2.axes.contourf(X[1], X[0], ub[1], 100)
                self.im2.autoscale()
                self.im3.set_UVC(ub[1]-(1-self.X[0]**2), ub[0])
                plt.pause(1e-6)
        '''
    def print_energy_and_divergence(self, t, tstep):
        if tstep % self.moderror == 0 and self.moderror > 0:
            if t==0:
                print (self.MOS.eigval)

            ub = self.u_.backward(self.ub)
            Bb = self.B_.backward(self.Bb)
            divu = self.divu().backward()
            divb = self.divb().backward()
            e3_u = dx(divu*divu)
            e3_b = dx(divb*divb)
            e0_u = self.e0_u
            e0_B = self.e0_B
            e1_u, e2_u, exact, e1_b, e2_b = self.compute_error(t)
            #print(e0_B, e1_b, e0_u, e1_u)
            self.acc_u[0] += abs(e1_u/e0_u-exact)*self.dt
            self.acc_B[0] += abs(e1_b/e0_B-exact)*self.dt
            if comm.Get_rank() == 0:
                #print("Time %2.5f Norms %2.12e %2.12e %2.12e %2.12e %2.12e %2.12e %2.12e %2.12e %2.12e %2.12e %2.12e %2.12e %2.12e" %(t, e1_u/e0_u, exact, e1_u/e0_u-exact, np.sqrt(e2_u), np.sqrt(e3_u),e1_b/e0_B, e1_b/e0_B-exact, np.sqrt(e2_b), np.sqrt(e3_b), 
        #inner(1, ub[0]*ub[0]), inner(1,ub[1]*ub[1]), inner(1,Bb[0]*Bb[0]), inner(1,Bb[1]*Bb[1])))
                print("Time %2.5f Norms %2.12e %2.12e %2.12e %2.12e %2.12e %2.12e %2.12e %2.12e %2.12e" %(t, e1_u/e0_u, exact, (e1_u)/(e0_u)-exact, np.sqrt(e2_u), np.sqrt(e3_u),e1_b/e0_B, e1_b, np.sqrt(e2_b), np.sqrt(e3_b), 
        ))

if __name__ == '__main__':
    from time import time
    from mpi4py_fft import generate_xdmf
    t0 = time()
    N = (128, 32)
    timestepper = 'IMEXRK222'
    #config['optimization']['mode'] = 'numba'
    d = {
        'N': N,
        'alpha': 1,
        'Re': 10000.,
        'Rem': 0.1,
        'B_0': (0, 0.2),
        'method': 'G',
        'dt': 0.01,
        'filename': f'KMM_OS_{N[0]}_{N[1]}',
        'conv': 0,
        'modplot': 100,
        'modsave': 1000,
        'moderror': 10,
        'family': 'C',
        'checkpoint': 10000000,
        'padding_factor': (1,1.5),
        'timestepper': timestepper,
        }
    MOS = True
    c = OrrSommerfeld(**d)
    t, tstep = c.initialize(from_checkpoint=False)
    c.solve(t=t, tstep=tstep, end_time=1)
    print('Computing time %2.4f'%(time()-t0))
    if comm.Get_rank() == 0:
        generate_xdmf('_'.join((d['filename'], 'U'))+'.h5')
