"""
Solve the Orr-Sommerfeld eigenvalue problem for MHD flow in a channel.
"""
import warnings
from scipy.linalg import eig
import numpy as np
import sympy as sp
from shenfun import FunctionSpace, Function, Dx, inner, TestFunction, \
    TrialFunction, MixedFunctionSpace, BlockMatrix

np.seterr(divide='ignore')

try:
    from matplotlib import pyplot as plt

except ImportError:
    warnings.warn("matplotlib not installed")

x = sp.Symbol('x', real=True)

class OrrSommerfeldMHD:
    def __init__(self, alpha=1., Re=8000., Rem=0.1, By=1., N=80, quad='GC', test='G', trial='G', **kwargs):
        kwargs.update(dict(alpha=alpha, Re=Re, Rem=Rem, By=By, N=N, quad=quad, test=test, trial=trial))
        vars(self).update(kwargs)
        self.x, self.w = None, None

    def interp(self, y, eigvals, eigvectors, eigval=1, verbose=False):
        """Interpolate solution eigenvector and it's derivative onto y

        Parameters
        ----------
            y : array
                Interpolation points
            eigvals : array
                All computed eigenvalues
            eigvectors : array
                All computed eigenvectors
            eigval : int, optional
                The chosen eigenvalue, ranked with descending imaginary
                part. The largest imaginary part is 1, the second
                largest is 2, etc.
            verbose : bool, optional
                Print information or not
        """
        nx, eigval = self.get_eigval(eigval, eigvals, verbose)
        SB, SD = self.get_trialspaces(self.trial, dtype='D')
        phi_hat_u = Function(SB)
        phi_hat_u[:-4] = np.squeeze(eigvectors[:self.N-4, nx])
        phi_u = phi_hat_u.eval(y)
        dphidy_u = Dx(phi_hat_u, 0, 1).eval(y)
        phi_hat_b = Function(SD)
        phi_hat_b[:-2] = np.squeeze(eigvectors[self.N-4:, nx])
        phi_b = phi_hat_b.eval(y)
        dphidy_b = Dx(phi_hat_b, 0, 1).eval(y)
        return eigval, phi_u, dphidy_u, phi_b, dphidy_b 

    def get_trialspaces(self, trial, dtype='d'):
        if trial == 'G':
            return (FunctionSpace(self.N, 'C', basis='ShenBiharmonic', quad=self.quad, dtype=dtype),#,FunctionSpace(self.N, 'C', basis='ShenBiharmonic', quad=self.quad, dtype=dtype))
                    FunctionSpace(self.N, 'C', basis='ShenDirichlet', quad=self.quad, dtype=dtype))
        return (FunctionSpace(self.N, 'C', basis='Phi2', quad=self.quad, dtype=dtype),
                FunctionSpace(self.N, 'C', basis='Phi1', quad=self.quad, dtype=dtype))

    def get_testspaces(self, trial):
        return trial.get_testspace(self.test)

    def assemble(self, scale=None):
        SB, SD = self.get_trialspaces(self.trial)
        testB = SB.get_testspace(self.test)
        testD = SD.get_testspace(self.test)
        TT = MixedFunctionSpace([SB, SD])
        TG = MixedFunctionSpace([testB, testD])
        v, q = TestFunction(TG)
        u, b = TrialFunction(TT)

        Re = self.Re
        Rem = self.Rem
        a = self.alpha
        By = self.By
        B1 = inner(-1j*Re*a*(Dx(u, 0, 2) - a**2*u), v)
        B4 = inner(-1j*a*Rem*b, q)
        A10 = inner(Dx(u, 0, 4) - 2*a**2*Dx(u, 0, 2) + (a**4-2*a*Re*1j)*u, v)
        A11 = inner(-1j*a*Re*(Dx(u, 0, 2)-a**2*u), (1-x**2)*v)
        A2 = inner(1j*a*Re*By*(Dx(b, 0, 2)-a**2*b), v)
        A3 = inner(1j*a*Rem*By*u, q)
        A40 = inner(Dx(b, 0, 2) - a**2*b, q)
        A41 = inner(-1j*a*Rem*b, (1-x**2)*q)
        A = BlockMatrix([A10, A11, A2, A3, A40, A41])
        B = BlockMatrix([B1, B4])
        AA, BB = A.diags().toarray(), B.diags().toarray()
        if scale is not None:
            assert isinstance(scale, tuple)
            assert len(scale) == 2
            N = self.N
            k0, k1 = scale
            assert len(k0) == 2
            assert len(k1) == 2
            # (0, 0)
            k = np.arange(N-4)
            j = np.arange(N-2)
            testp = 1/(k+1)**(-k0[0]) if k0[0] < 0 else (k+1)**k0[0]
            trialp = 1/(k+1)**(-k1[0]) if k1[0] < 0 else (k+1)**k1[0]
            d =  testp[:, None] * trialp[None, :]
            AA[:N-4, :N-4] *= d
            BB[:N-4, :N-4] *= d
            # (0, 1)
            testp = 1/(k+1)**(-k0[0]) if k0[0] < 0 else (k+1)**k0[0]
            trialp = 1/(j+1)**(-k1[1]) if k1[1] < 0 else (j+1)**k1[1]
            d =  testp[:, None] * trialp[None, :]
            AA[:N-4, N-4:] *= d
            BB[:N-4, N-4:] *= d
            # (1, 0)
            testp = 1/(j+1)**(-k0[1]) if k0[1] < 0 else (j+1)**k0[1]
            trialp = 1/(k+1)**(-k1[0]) if k1[0] < 0 else (k+1)**k1[0]
            d =  testp[:, None] * trialp[None, :]
            AA[N-4:, :N-4] *= d
            BB[N-4:, :N-4] *= d
            # (1, 1)
            testp = 1/(j+1)**(-k0[1]) if k0[1] < 0 else (j+1)**k0[1]
            trialp = 1/(j+1)**(-k1[1]) if k1[1] < 0 else (j+1)**k1[1]
            d =  testp[:, None] * trialp[None, :]
            AA[N-4:, N-4:] *= d
            BB[N-4:, N-4:] *= d
            d = (1/np.sqrt(AA.diagonal()))[None, :] * (1/np.sqrt(AA.diagonal()))[:, None]
            AA *= d
            BB *= d
        return AA, BB

    def solve(self, verbose=False):
        """
        Solve the coupled Orr-Sommerfeld and induction equations
        """
        if verbose:
            print('Solving the Orr-Sommerfeld and induction eigenvalue problem...')
            print('Re = '+str(self.Re)+' and alpha = '+str(self.alpha))
        A, B = self.assemble()
        return eig(A, B)

    @staticmethod
    def get_eigval(nx, eigvals, verbose=False):
        """
        Get the chosen eigenvalue

        Parameters
        ----------
            nx : int
                The chosen eigenvalue. nx=1 corresponds to the one with the
                largest imaginary part, nx=2 the second largest etc.
            eigvals : array
                Computed eigenvalues
            verbose : bool, optional
                Print the value of the chosen eigenvalue. Default is False.
        """
        indices = np.argsort(np.imag(eigvals))
        indi = indices[-1*np.array(nx)]
        eigval = eigvals[indi]
        if verbose:
            ev = list(eigval) if np.ndim(eigval) else [eigval]
            indi = list(indi) if np.ndim(indi) else [indi]
            for i, (e, v) in enumerate(zip(ev, indi)):
                print('Eigenvalue {} ({}) = {:2.16e}'.format(i+1, v, e))
        return indi, eigval

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Orr Sommerfeld parameters')
    parser.add_argument('--N', type=int, default=120,
                        help='Number of discretization points')
    parser.add_argument('--Re', default=10000.0, type=float,
                        help='Reynolds number')
    parser.add_argument('--Rem', default=0.1, type=float,
                        help='Magnetic Reynolds number')
    parser.add_argument('--alpha', default=1.0, type=float,
                        help='Parameter')
    parser.add_argument('--By', default=0.2, type=float,
                        help='Parameter')
    parser.add_argument('--test', default='G', type=str,
                        help='G or PG, Galerkin or Petrov-Galerkin')
    parser.add_argument('--trial', default='G', type=str,
                        help='G or PG, Galerkin or Petrov-Galerkin')
    parser.add_argument('--quad', default='GC', type=str, choices=('GC', 'GL', 'LG'),
                        help='Discretization points: GC: Gauss-Chebyshev, GL: Gauss-Lobatto')
    parser.add_argument('--plot', dest='plot', action='store_true', help='Plot eigenvalues')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='Print results')
    parser.set_defaults(plot=False)
    parser.set_defaults(verbose=False)
    args = parser.parse_args()
    z = OrrSommerfeldMHD(**vars(args))
    evals, evectors = z.solve(args.verbose)
    d = z.get_eigval(1, evals, args.verbose)

    if args.plot:
        plt.figure()
        evi = evals*z.alpha
        plt.plot(evi.imag, evi.real, 'o')
        plt.axis([-10, 0.1, 0, 1])
        plt.show()
