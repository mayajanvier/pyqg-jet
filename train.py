import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from qg import QGM_differentiable, Unet, HybridForecaster
import json

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

def train_single_step(param, is_augmented=True, num_steps=10, integration_method="heun2"):
    # init hybrid model
    qg_multilayer = QGM_differentiable(param)
    nb_channels = param["nl"] * 2  # 2 channels per layer (dq, dp)
    unet = Unet(nb_channels, nb_channels, param["n_ens"]).to(param["device"])
    net = HybridForecaster(
        model_phy=qg_multilayer,
        model_aug=unet,
        dt=param["dt"],
        num_steps=num_steps, 
        integration_method=integration_method,
        is_augmented=is_augmented,
    )
    loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    nb_epochs=10

    # init
    if param['nx'] == 97: # LR and original paper
        p = torch.from_numpy(np.load('./p_380yrs_HRDS.npy')).to(param['device'])
    else: # TODO generate initial state of right size 
        p = qg_multilayer.p0    
    q_over_f0 = qg_multilayer.compute_q_over_f0_from_p(p)
    y0 = torch.stack((q_over_f0,p),0).to(param["device"]) # nc nl nx ny
    t = torch.arange(0, (net.num_steps+1) * net.dt, net.dt).type(torch.float32).to(param["device"]) # time steps
    for k in range(nb_epochs):
        optimizer.zero_grad()
        out = net(y0, t)
        q_over_f0, p = out[:,0], out[:,1]
        q = qg_multilayer.f0 * q_over_f0 
        
        # compute loss
        loss_value = loss(p, q)
        # backprop
        loss_value.backward(retain_graph=True)
        optimizer.step()
        print(f'Epoch {k}, Loss {loss_value}')
    
    return None 


# to compare to original one
def run_QG(param, num_steps=10, integration_method="heun2", zero_init=False):
    # init hybrid model
    dt = param["dt"]
    qg_multilayer = QGM_differentiable(param)
    nb_channels = param["nl"] * 2  # 2 channels per layer (dq, dp)
    unet = Unet(nb_channels, nb_channels, param["n_ens"]).to(param["device"])
    net = HybridForecaster(
        model_phy=qg_multilayer,
        model_aug=unet,
        dt=dt,
        num_steps=1, 
        integration_method=integration_method,
        is_augmented=False,
    )

    # init
    if zero_init:
        p = qg_multilayer.p0 
    else:
        if param['nx'] == 97: # LR and original paper
            p = torch.from_numpy(np.load('./p_380yrs_HRDS.npy')).to(param['device']) 
        else:
            p = qg_multilayer.p0
    q_over_f0 = qg_multilayer.compute_q_over_f0_from_p(p)
    y0 = torch.stack((q_over_f0,p),0).to(param["device"]) # nc nl nx ny
    t = torch.arange(0, (net.num_steps+1)*dt, dt).type(torch.float32).to(param["device"]) # single step 
    # t_all = torch.arange(0, num_steps+1, 1).type(torch.float32).to(param["device"]) # all steps 
    # out_all = net(y0, t_all)
    for k in range(num_steps):
        out = net(y0, t)
        y0 = out[-1]
        q_over_f0, p = out[:,0], out[:,1]
        q = qg_multilayer.f0 * q_over_f0 
    final_state = out[-1]
    #print(final_state == out_all[-1])
    return final_state


def run_save_ground_truth(param_path, n_years, integration_method="heun2", save_every_yr=1, folder='data/HR'):
    # open params json file
    with open(param_path, 'r') as f:
        param = json.load(f)
    # init hybrid model
    dt = param["dt"]
    n_steps = int(n_years*365*24*3600 / dt)
    #n_steps_every_yr = 10  # test
    n_steps_every_yr = int(save_every_yr*365*24*3600 / dt)
    assert n_steps % n_steps_every_yr == 0
    nb_steps = n_steps // n_steps_every_yr

    qg_multilayer = QGM_differentiable(param)
    net = HybridForecaster(
        model_phy=qg_multilayer,
        model_aug=None,
        dt=dt,
        num_steps=n_steps_every_yr, # one year step
        integration_method=integration_method,
        is_augmented=False, # QG only 
    )

    # init
    p = qg_multilayer.p0 
    q_over_f0 = qg_multilayer.compute_q_over_f0_from_p(p)
    y0 = torch.stack((q_over_f0,p),0).to(param["device"]) # nc nl nx ny
    t = torch.arange(0, (net.num_steps+1)*dt, dt).type(torch.float32).to(param["device"]) # single step 
    for k in range(nb_steps):
        out = net(y0, t) # single year step
        y0 = out[-1]
        np.save(f'./{folder}/y_{k}yrs.npy', y0.cpu().numpy())
    # save final state
    np.save(f'./{folder}/y_{k}yrs.npy', y0.cpu().numpy())

if __name__ == '__main__':
    #qg_only()
    torch.autograd.set_detect_anomaly(True)
    param_aug = {
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
    #train_single_step(param_aug, is_augmented=True, num_steps=10, integration_method="heun2")

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
    #run_QG(param, num_steps=10, integration_method="heun2")

    run_save_ground_truth("parameters/HR_params.json", n_years=1)