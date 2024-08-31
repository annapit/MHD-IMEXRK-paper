#import matplotlib.pyplot as plt
from shenfun import *
from ChannelFlow2D import KMM


class OrrSommerfeld(KMM):

    def __init__(self, N=(32, 32), domain=((-1, 1), (0, 2*np.pi)), Re=8000., alpha=1., method = "G",
                 dt=0.1, conv=0, modplot=100, modsave=1e8, moderror=100, filename='KMM',
                 family='C', padding_factor=(1, 1.5), checkpoint=1000, timestepper='IMEXRK3'):
        KMM.__init__(self, N=N, domain=(domain[0], (domain[1][0], domain[1][1]/alpha)), nu=1/Re, dt=dt, conv=conv, modplot=modplot,
                     modsave=modsave, moderror=moderror, filename=filename, family=family,
                     padding_factor=padding_factor, checkpoint=checkpoint, timestepper=timestepper,
                     dpdy=-2/Re)
        self.Re = Re
        self.alpha = alpha
        self.method = method
        self.filename = filename

        # Make a text file for saving the energy and divergence
        with open(f"{self.filename}.txt", "w") as f:
            f.write(f"{'Time'::^11} {'Exact':^11} {'u_error':^11} {'u_error_exact':^11} {'u_norm':^11} {'u_div':^11} {'u_norm_x':^11} {'u_norm_y':^11} {'u_acc':^11}\n")


    def initialize(self, from_checkpoint=False):
        if from_checkpoint:
            return self.init_from_checkpoint()

        from OrrSommerfeld_eigs import OrrSommerfeld
        self.OS = OS = OrrSommerfeld(Re=self.Re, N=128, alpha=self.alpha, test=self.method, trial=self.method)
        eigvals, eigvectors = OS.solve(False)
        OS.eigvals, OS.eigvectors = eigvals, eigvectors
        self.initOS(OS, eigvals, eigvectors, self.ub)
        self.u_ = self.ub.forward(self.u_)
        self.e0 = 0.5*dx(self.ub[0]**2+(self.ub[1]-(1-self.X[0]**2))**2)
        self.acc = np.zeros(1)
        self.print_energy_and_divergence(0, 0)
        return 0, 0

    def initOS(self, OS, eigvals, eigvectors, U, t=0.):
        X = self.X
        x = X[0][:, 0].copy()
        eigval, phi, dphidy = OS.interp(x, eigvals, eigvectors, eigval=1, verbose=False)
        OS.eigval = eigval
        for j in range(U.shape[2]):
            y = X[1][0, j]
            v = (1-x**2) + 1e-5*np.real(dphidy*np.exp(1j*self.alpha*(y-eigval*t)))
            u = -1e-5*np.real(1j*self.alpha*phi*np.exp(1j*self.alpha*(y-eigval*t)))
            U[0, :, j] = u
            U[1, :, j] = v

    def compute_error(self, t):
        ub = self.u_.backward(self.ub)
        pert = (ub[1] - (1-self.X[0]**2))**2 + ub[0]**2
        e1 = 0.5*dx(pert)
        exact = np.exp(2*np.imag(self.alpha*self.OS.eigval)*t)
        U0 = self.work[(ub, 0, True)]
        self.initOS(self.OS, self.OS.eigvals, self.OS.eigvectors, U0, t=t)
        pert = (ub[0] - U0[0])**2 + (ub[1] - U0[1])**2
        e2 = 0.5*dx(pert)
        return e1, e2, exact

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
                self.im1.axes.contourf(X[1], X[0], ub[0], 100)
                self.im1.autoscale()
                self.im2.axes.contourf(X[1], X[0], ub[1], 100)
                self.im2.autoscale()
                self.im3.set_UVC(ub[1]-(1-self.X[0]**2), ub[0])
                plt.pause(1e-6)
        '''
    def print_energy_and_divergence(self, t, tstep):
        if tstep % self.moderror == 0 and self.moderror > 0:
            if t==0:
                print (self.OS.eigval)
            ub = self.u_.backward(self.ub)
            divu = self.divu().backward()
            e3 = dx(divu*divu)
            e0 = self.e0
            e1, e2, exact = self.compute_error(t)
            self.acc[0] += abs(e1/e0-exact)*self.dt
            if comm.Get_rank() == 0:
                print("Time %2.5f Norms %2.12e %2.12e %2.12e %2.12e %2.12e" %(t, exact, e1/e0, e1/e0-exact, np.sqrt(e2), np.sqrt(e3)))
                # Save energy and divergence to file
                with open(f"{self.filename}.txt", "a") as f:
                    f.write(f"{t:2.5f} {exact:2.12e} {e1/e0:2.12e} {e1/e0-exact:2.12e} {np.sqrt(e2):2.12e} {np.sqrt(e3):2.12e} {inner(1, ub[0]*ub[0]):2.12e} {inner(1,ub[1]*ub[1]):2.12e} {self.acc[0]:2.12e}\n")

if __name__ == '__main__':
    from time import time
    import argparse
    t0 = time()
    parser = argparse.ArgumentParser(description='OrrSommerfeld2D')
    parser.add_argument('--N', type=int, nargs=2, default=[128, 32], help='Number of quadrature points in each direction')
    parser.add_argument('--Re', type=float, default=10000., help='Reynolds number')
    parser.add_argument('--dt', type=float, default=0.01, help='Time step size')
    parser.add_argument('--conv', type=int, default=0, help='Convergence test')
    parser.add_argument('--alpha', type=float, default=1., help='Alpha value')
    parser.add_argument('--method', type=str, default='G', help='Method')
    parser.add_argument('--modplot', type=int, default=-1, help='Plotting frequency')
    parser.add_argument('--modsave', type=int, default=-1, help='Saving frequency')
    parser.add_argument('--family', type=str, default='C', help='Family type')
    parser.add_argument('--checkpoint', type=int, default=10000000, help='Checkpoint frequency')
    parser.add_argument('--padding_factor', type=float, default=1., help='Padding factor')
    parser.add_argument('--timestepper', type=str, default='IMEXRK443', help='Time stepper type')

    args = parser.parse_args()

    d = {
        'N': args.N,
        'Re': args.Re,
        'dt': args.dt,
        'filename': f"KMM_OS_N_{args.N[0]}_{args.N[1]}_alpha_{args.alpha}_Re_{args.Re}_dt_{args.dt}_{args.timestepper}_{args.method}",
        'conv': args.conv,
        'alpha': args.alpha,
        'method': args.method,
        'modplot': args.modplot,
        'modsave': args.modsave,
        'moderror': int(1/(10*args.dt)),
        'family': args.family,
        'checkpoint': args.checkpoint,
        'padding_factor': args.padding_factor,
        'timestepper': args.timestepper
    }
    print(d['filename'])
    OS = True
    c = OrrSommerfeld(**d)
    t, tstep = c.initialize(from_checkpoint=False)
    c.solve(t=t, tstep=tstep, end_time=100)
    print('Computing time %2.4f'%(time()-t0))
    with open(f"{d['filename']}.txt", "a") as f:
        f.write(f"Computing time {time()-t0}\n")
        f.write(f"OS eigenvalue: {c.OS.eigval}\n")
    cleanup(vars(c))
