from shenfun import *
import matplotlib.pyplot as plt
import numpy as np
import sympy
import warnings
warnings.filterwarnings('ignore')

x, y = sympy.symbols("x,y", real=True)

class Burgers:
    """Class for solving the Burgers equation with IMEXRK time scheme

    Parameters
    ----------
    N : int
        Number of basis functions
    nu : number, optional
        The diffusion coefficient
    dt : number, optional
        timestep
    domain : 2-tuple of numbers
        Size of the domain
    u0 : Sympy function of x
        Used for specifying initial condition
    kind : str, optional
        'G' or 'PG'
    family : str, optional
        'L' or 'C'
    padding_factor': number
        Used for calculation of convection (convolution theorem)
    xi : linspace
        Used to evaluate solution in specified mesh points
    timestepper : str
        Choose between 'IMEXRK111', 'IMEXRK222' or 'IMEXRK443'
    """
    def __init__(self,
                 N=32,
                 nu=3.23739,
                 dt=0.001,
                 domain=(-1, 1),
                 u0 = sympy.sin(sympy.pi*x),
                 kind = "G",
                 family='C',
                 padding_factor=1.5,
                 modplot=100,
                 xi = np.linspace(0,1,100),
                 timestepper='IMEXRK111'):
        self.N = N
        self.nu = nu
        self.dt = dt
        self.u0 = u0
        self.bc = (self.u0.subs(x, domain[0]).n(), self.u0.subs(x, domain[1]).n())
        self.kind = kind
        self.family = family
        self.modplot = modplot
        self.xi = xi
        self.im1 = None
        self.padding_factor = padding_factor
        self.PDE = PDE = globals().get(timestepper)

        self.V = FunctionSpace(self.N, self.family, domain=domain, bc=self.bc)
        self.T = self.V.get_orthogonal()

        testspace = self.V.get_testspace(self.kind)
        v = TestFunction(testspace)
        sol2 = chebyshev.la.Helmholtz if self.kind == 'G' else la.Solver

        self.u_ = Function(self.V)
        self.Nu_ = Function(self.V)      # Convection nabla(u) dot u
        self.dudx = Project(Dx(self.u_, 0, 1), self.T)

        self.pdes = {
        'u' :PDE(v,
                 self.u_,
                 lambda f: self.nu*div(grad(f)),
                 -Expr(self.Nu_),
                 dt=self.dt,
                 solver=sol2,
                 )
                 }

    def convection(self):
        up = self.u_.backward(padding_factor=self.padding_factor)
        dudxp = self.dudx().backward(padding_factor=self.padding_factor)
        self.Nu_ = up.function_space().forward(up*dudxp, self.Nu_)


    def initialize(self):
        self.u_[:] = project(self.u0, self.V)
        return 0, 0

    def prepare_step(self, rk):
        self.convection()

    def mesh(self):
        return self.V.mesh(kind='uniform')

    def assemble(self):
        for pde in self.pdes.values():
            pde.assemble()

    def init_plots(self):
        plt.figure(1, figsize=(6, 3))
        self.im1 = plt.plot(self.mesh(), self.u_.backward(mesh='uniform'))
        plt.draw()
        plt.pause(1e-6)

    def update(self, t, tstep):
        self.plot(t, tstep)
    
    def plot(self, t, tstep):
        if self.im1 is None and self.modplot > 0:
            self.init_plots()

        if tstep % self.modplot == 0 and self.modplot > 0:
            plt.figure(1)
            self.im1.clear()
            plt.clf()
            self.im1 = plt.plot(self.mesh(), self.u_.backward(mesh='uniform'))
            plt.pause(1e-6)

    def solve(self, t=0, tstep=0, end_time=1000, save_step=100):
        plotdata = {0: self.u_.eval(self.xi)}

        self.assemble()

        while t < end_time-1e-8:
            for rk in range(self.PDE.steps()):
                self.prepare_step(rk)
                for eq in self.pdes.values():
                    eq.compute_rhs(rk)
                for eq in self.pdes.values():
                    eq.solve_step(rk)
            t += self.dt
            tstep += 1
            #self.update(t, tstep)           
            if tstep % save_step == 0: # save every save_step timestep
                u = self.u_.eval(self.xi)
                plotdata["{:.2f}".format(t)] = u.copy()
                
        return plotdata


if __name__ == '__main__':
    from time import time
    t0 = time()
    N = 32
    timestepper = 'IMEXRK222'
    dt= 1e-2
    kind ='G'
    nu= 0.01
    d = {'N': N,
        'nu': nu,
        'dt': dt,
        'u0': sympy.sin(sympy.pi*x),
        'kind': kind,
        'domain': (0, 1),
        'family': 'C',
        'padding_factor': 1,
        'modplot': 10,
        'xi' : np.linspace(0,1,100),
        'timestepper': timestepper
        }
    c = Burgers(**d)
    t, tstep = c.initialize()
    data = c.solve(t=t, tstep=tstep, end_time=1, save_step=int(1/(10*dt)))
    print('Computing time %2.4f'%(time()-t0))
