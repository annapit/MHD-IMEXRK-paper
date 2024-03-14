import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy import sparse
from shenfun import *

x, t, c, L = sp.symbols('x,t,c,L')

def plot_with_offset(data, xj, figsize=(4, 3)):
    Nd = len(data)
    v = np.array(list(data.values()))
    t = np.array(list(data.keys()))
    dt = t[1]-t[0]
    v0 = abs(v).max()
    fig = plt.figure(facecolor='k', figsize=figsize)
    ax = fig.add_subplot(111, facecolor='k')
    for i, u in data.items():
        ax.plot(xj, u+i*v0/dt, 'w', lw=2, zorder=i)
        ax.fill_between(xj, u+i*v0/dt, i*v0/dt, facecolor='k', lw=0, zorder=i-1)
    plt.show()

class Diffusion:
    """Class for solving the diffusion equation

    Parameters
    ----------
    N : int
        Number of basis functions
    L0 : number
        The extent of the domain, which is [0, L0]
    nu : number, optional
        The diffusion coefficient
    dt : number, optional
        timestep
    bc : 2-tuple of numbers
        The Dirichlet boundary conditions at the two edges
    u0 : Sympy function of x, t, c and L
        Used for specifying initial condition
    kind : str, optional
        'G' or 'PG'
    family : str, optional
        'L' or 'C'
    """
    def __init__(self, N, L0=1, nu=1, dt=1, u0=sp.cos(sp.pi*x/L)+sp.cos(10*sp.pi*x/L), kind='G', family='L'):
        self.N = N
        self.L = L0
        self.nu = nu
        self.dt = dt
        self.u0 = u0.subs({L: L0})
        self.bc = (self.u0.subs(x, 0).n(), self.u0.subs(x, L0).n())
        self.kind = kind
        self.family = family
        self.V = self.get_function_space()
        self.uh_np1 = Function(self.V)
        self.uh_n = Function(self.V)
        self.A, self.S = self.get_mats()
        self.A = self.A[0]
        self.lu = la.Solver(self.A)
        self.A = self.A.diags('csr')
        self.S = self.S.diags('csr')

    def get_function_space(self):
        return FunctionSpace(self.N, self.family, domain=(0, self.L), bc=self.bc)

    def get_mats(self):
        """Return mass and stiffness matrices
        """
        N = self.N
        u = TrialFunction(self.V)
        testspace = self.V.get_testspace(self.kind)
        v = TestFunction(testspace)
        a = inner(u, v)
        s = inner(-div(grad(u)), v)
        return a, s

    def get_max_dt(self):
        return 2/abs(np.linalg.eig(np.linalg.inv(sol.A.toarray()) @ sol.S.toarray())[0]).max()

    def mesh(self):
        return self.V.mesh(kind='uniform')

    def __call__(self, Nt, dt=None, save_step=100):
        """Solve diffusion equation

        Parameters
        ----------
        Nt : int
            Number of time steps
        dt : number
            timestep
        save_step : int, optional
            Save solution every save_step time step

        Returns
        -------
        Dictionary with key, values as timestep, array of solution
        The number of items in the dictionary is Nt/save_step, and
        each value is an array of length N+1

        """
        # Initialize
        self.uh_n[:] = project(self.u0, self.V) # unm1 = u(x, 0)
        plotdata = {0: self.uh_n.backward(mesh='uniform').copy()}

        # Solve
        f = Function(self.V)
        for n in range(2, Nt+1):
            f[:-2] = self.A @ self.uh_n[:-2] - self.dt*self.nu*(self.S @ self.uh_n[:-2])
            self.uh_np1 = self.lu(f, self.uh_np1)
            self.uh_n[:] = self.uh_np1
            if n % save_step == 0: # save every save_step timestep
                plotdata[n] = self.uh_np1.backward(mesh='uniform').copy()
        return plotdata

if __name__ == '__main__':
    #dt = 2.1980578790345163e-05 # Leg 40 (2/np.linalg.eig(np.linalg.inv(sol.A.toarray()) @ sol.S.toarray())[0][0])
    #dt = 7.46871942608253e-05 # Leg PG 40
    #dt = 1.2332249161314773e-05 # Cheb 40
    dt = 3.734059255129313e-05 # Cheb PG 40
    # Med dt=dt skal det være stabilt. Med dt=dt*1.01 skal det divergere.
    sol = Diffusion(43, dt=dt, L0=2, nu=1, kind='PG', family='C')
    data = sol(1000, save_step=100)
    plot_with_offset(data, sol.mesh())
