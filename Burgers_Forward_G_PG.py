import numpy as np
import sympy
import matplotlib.pyplot as plt
from scipy import sparse
from shenfun import *

x, y = sympy.symbols("x,y", real=True)


class BurgersForward:
    """Class for solving the Burgers equation with Forward Euler time scheme

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
                 xi = np.linspace(0,1,100)):
        self.N = N
        self.nu = nu
        self.dt = dt
        self.u0 = u0
        self.domain = domain
        self.bc = (self.u0.subs(x, self.domain[0]).n(), self.u0.subs(x, self.domain[1]).n())
        self.kind = kind
        self.family = family
        self.modplot = modplot
        self.xi = xi
        self.im1 = None
        self.padding_factor = padding_factor
        self.V = FunctionSpace(self.N, self.family, domain=self.domain, bc=self.bc)
        self.T = self.V.get_orthogonal()

        self.uh_np1 = Function(self.V)
        self.uh_n = Function(self.V)    
        self.Nu = Function(self.V)      # Convection nabla(u) dot u
        self.dudx = Project(Dx(self.uh_n, 0, 1), self.T)

        u = TrialFunction(self.V)
        self.Test = self.V.get_testspace(self.kind)
        v = TestFunction(self.Test)
        

        self.A = inner(u, v)
        self.K = inner(Expr(self.Nu), v)
        self.S = inner(-div(grad(u)), v)

        self.lu = la.Solver(self.A)
        self.A = self.A.diags('csr')
        self.S = self.S.diags('csr')

    def convection(self):
        up = self.uh_n.backward(padding_factor=self.padding_factor)
        dudxp = self.dudx().backward(padding_factor=self.padding_factor)
        self.Nu = up.function_space().forward(up*dudxp, self.Nu)
        self.K = inner(Expr(self.Nu), TestFunction(self.Test))

    def mesh(self):
        return self.V.mesh(kind='uniform')
    
    def initialize(self):
        self.uh_n[:] = project(self.u0, self.V)
        return 0, 0
    
    def init_plots(self):
        plt.figure(1, figsize=(6, 3))
        self.im1 = plt.plot(self.mesh(), self.uh_np1.backward(mesh='uniform'))
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
            self.im1 = plt.plot(self.mesh(), self.uh_np1.backward(mesh='uniform'))
            plt.pause(1e-6)

    def solve(self, t=0, tstep=0, end_time=1, save_step=100):
        plotdata = {0: self.uh_n.eval(self.xi)}

        # Solve
        f = Function(self.V)
        Nt = int((end_time-t)/self.dt)
        for n in range(2, Nt+1):
            self.convection()
            f[:-2] = self.A @ self.uh_n[:-2] - self.dt*self.nu*(self.S @ self.uh_n[:-2]) - self.dt*(self.K[:self.Test.dim()])
            self.uh_np1 = self.lu(f, self.uh_np1)
            self.uh_n[:] = self.uh_np1
            t += self.dt
            #self.update(t, n) 
            if n % save_step == 0: # save every save_step timestep
                u = self.uh_np1.eval(self.xi)
                plotdata["{:.2f}".format(t)] = u.copy()

        return plotdata

if __name__ == '__main__':
    from time import time
    t0 = time()
    N = 16
    dt= 1/500
    kind ='PG'
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
        'xi' : np.linspace(0,1,100)
        }
    c = BurgersForward(**d)
    t, tstep = c.initialize()
    data = c.solve(t=t, tstep=tstep, end_time=1, save_step=int(1/(10*dt)))
    print('Computing time %2.4f'%(time()-t0))
