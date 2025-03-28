import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from qg import QGM

torch.backends.cudnn.deterministic = True

def qg_only():
    param = {
        # 'nx': 769, # HR
        # 'ny': 961, # HR
        'nx': 97, # LR
        'ny': 121, # LR
        'Lx': 3840.0e3, # Length in the x direction (m)
        'Ly': 4800.0e3, # Length in the y direction (m)
        'nl': 3, # number of layers
        'heights': [350., 750., 2900.], # heights between layers (m)
        'reduced_gravities': [0.025, 0.0125], # reduced gravity numbers (m/s^2)
        'f0': 9.375e-5, # coriolis (s^-1)
        'a_2': 0., # laplacian diffusion coef (m^2/s)
        # 'a_4': 2.0e9, # HR
        'a_4': 5.0e11, # LR
        'beta': 1.754e-11, # coriolis gradient (m^-1 s^-1)
        'delta_ek': 2.0, # eckman height (m)
        # 'dt': 600., # HR
        'dt': 1200., # LR
        'bcco': 0.2, # boundary condition coef. (non-dim.)
        'tau0': 2.0e-5, # wind stress magnitude m/s^2
        'n_ens': 0, # 0 for no ensemble,
        'device': 'cpu', # torch only, 'cuda' or 'cpu'
        'p_prime': ''
    }

    import time
    qg_multilayer = QGM(param)

    if param['nx'] == 97: # LR
        qg_multilayer.p = torch.from_numpy(np.load('./p_380yrs_HRDS.npy')).to(param['device'])
    qg_multilayer.compute_q_over_f0_from_p()


    # time params
    dt = param['dt']
    t = 0

    freq_plot = 1000 # LR
    # freq_plot = 50 # HR
    freq_checknan = 10000
    freq_log = 1000
    n_years = 2
    n_steps = int(n_years*365*24*3600 / dt)

    if freq_plot > 0: 
        plt.ion()
        plt.figure()
        f,a = plt.subplots(1,2)
        u = (qg_multilayer.compute_u()[0]).cpu().numpy()
        um, uM = -1.1*np.abs(u).max(), 1.1*np.abs(u).max()
        im = a[0].imshow(u[0].T, cmap='bwr', origin='lower', vmin=um, vmax=uM, animated=True)
        a[0].set_title('zonal velocity')
        f.colorbar(im, ax=a[0])
        q = (qg_multilayer.q_over_f0*qg_multilayer.f0).cpu().numpy()
        qm, qM = -1.1*np.abs(q).max(), 1.1*np.abs(q).max()
        im = a[1].imshow(q[0].T, cmap='bwr', origin='lower', vmin=qm, vmax=qM, animated=True)
        a[1].set_title('potential vorticity')
        f.colorbar(im, ax=a[1])
        plt.pause(5)
        plt.show()

    times, outputs = [], []

    t0 = time.time()
    for n in range(1, n_steps+1):
        qg_multilayer.step()
        t += dt

        if n % freq_checknan == 0 and torch.isnan(qg_multilayer.p).any():
            raise ValueError('Stopping, NAN number in p at iteration {n}.')

        if freq_plot > 0 and n % freq_plot == 0:
            u = (qg_multilayer.compute_u()[0]).cpu().numpy()
            a[0].imshow(u[0].T, cmap='bwr', origin='lower', vmin=um, vmax=uM, animated=True)
            q = (qg_multilayer.q_over_f0*qg_multilayer.f0).cpu().numpy() # q =q/f0 * f0
            a[1].imshow(q[0].T, cmap='bwr', origin='lower', vmin=qm, vmax=qM, animated=True)
            plt.suptitle(f't={t/(365*24*3600):.2f} years.')
            plt.pause(0.1)

        if freq_log > 0 and n % freq_log == 0:
            q, p = (qg_multilayer.f0 * qg_multilayer.q_over_f0).cpu().numpy(), qg_multilayer.p.cpu().numpy()
            print(f'{n=:06d}, t={t/(365*24*60**2):.2f} yr, ' \
                    f'p: ({p.mean():+.1E}, {np.abs(p).mean():.6E}), ' \
                    f'q: ({q.mean():+.1E}, {np.abs(q).mean():.6E}).')
    print(100*(time.time()-t0)/(60*60))
    plt.show()

def train_single_step():
    param = {
        # 'nx': 769, # HR
        # 'ny': 961, # HR
        #'nx': 97, # LR
        #'ny': 121, # LR
        'nx': 100, # LR
        'ny': 120, # LR
        'Lx': 3960.0e3, # Length in the x direction (m)
        'Ly': 4760.0e3, # Length in the y direction (m)
        'nl': 3, # number of layers
        'heights': [350., 750., 2900.], # heights between layers (m)
        'reduced_gravities': [0.025, 0.0125], # reduced gravity numbers (m/s^2)
        'f0': 9.375e-5, # coriolis (s^-1)
        'a_2': 0., # laplacian diffusion coef (m^2/s)
        # 'a_4': 2.0e9, # HR
        'a_4': 5.0e11, # LR
        'beta': 1.754e-11, # coriolis gradient (m^-1 s^-1)
        'delta_ek': 2.0, # eckman height (m)
        # 'dt': 600., # HR
        'dt': 1200., # LR
        'bcco': 0.2, # boundary condition coef. (non-dim.)
        'tau0': 2.0e-5, # wind stress magnitude m/s^2
        'n_ens': 0, # 0 for no ensemble,
        'device': 'cpu', # torch only, 'cuda' or 'cpu'
        'p_prime': '' # corrective pressure field -> our NN 
    }

    qg_multilayer = QGM(param)
    loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(qg_multilayer.unet.parameters(), lr=1e-3)
    nb_epochs=10
    for k in range(nb_epochs):
        qg_multilayer.step()
        # get q,p 
        u = qg_multilayer.compute_u()[0]
        q = qg_multilayer.q_over_f0*qg_multilayer.f0
        p = qg_multilayer.p
        # compute loss
        loss_value = loss(q, p)
        # backprop
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()
        print(f'Epoch {k}, Loss {loss_value}')
    
    return None 

if __name__ == '__main__':
    #qg_only()
    torch.autograd.set_detect_anomaly(True)
    train_single_step()