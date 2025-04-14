import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from operators import * 
from einops import rearrange
#from runge_kutta import RK_tableaux, RK_solver_fixed
from torchdiffeq import odeint_adjoint, odeint

class Unet(nn.Module):
    """ Inputs: size (nx,ny)=97x121 (LR) with 6 channels: (dq, dp) x 3 layers
    Architecture:
        - 2 decreasing resolution blocks, 2 increasing resolution blocks 
        with skip connections
        - Latent layers: 3x3 conv, BN, ReLU
        - Down blocks: twice 3x3 conv, BN, ReLU + 2x2 max pooling
        - Up blocks: bilinear upsampling+ twice 3x3 conv, BN, ReLU"""
    def __init__(self, in_channels, out_channels, n_ens=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_ens = n_ens
        # layers
        self.layer1_down = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            #nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            #nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.layer2_down = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            #nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            #nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.latent = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            #nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            #nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            #nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            #nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer3_up = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            #nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            #nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.layer4_up = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            #nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            #nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, out_channels, 3, padding=1),
            #nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
        # up and down scalers
        self.down1 = nn.MaxPool2d(2) # max pooling
        self.down2 = nn.MaxPool2d(2) # max pooling
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) #bilinear upsampling 
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) #bilinear upsampling

    def forward(self, x):
        # down
        x1 = self.layer1_down(x) # keep for skip connection
        x2 = self.down1(x1)
        x3 = self.layer2_down(x2) # keep for skip connection
        x4 = self.down2(x3)
        # latent
        x5 = self.latent(x4)
        # up
        x6 = self.up1(x5)
        x7 = self.layer3_up(x6 + x3) # skip connection
        x8 = self.up2(x7)
        x9 = self.layer4_up(x8+ x1) # skip connection
        if self.n_ens == 0:
            return x9[0,0:3,:,:], x9[0,3:6,:,:] # channels, nx, ny
        else:
            return x9[:,:,0:3,:,:], x9[:,:,3:6,:,:] # bs, n_ens, channels, nx, ny

# class QGM:
#     """Implementation of multilayer quasi-geostrophic model
#     in variables pressure p and potential vorticity q.
#     """

#     def __init__(self, param):
#         ### parameters of QG equations
#         self.nx = param['nx']
#         self.Lx = param['Lx']
#         self.ny = param['ny']
#         self.Ly = param['Ly']
#         self.nl = param['nl']
#         self.heights = param['heights']
#         self.reduced_gravities = param['reduced_gravities']
#         self.f0 = param['f0']
#         self.a_2 = param['a_2']
#         self.a_4 = param['a_4']
#         self.beta = param['beta']
#         self.delta_ek = param['delta_ek']
#         self.dt = param['dt']
#         self.bcco = param['bcco']
#         self.n_ens = param['n_ens']
#         self.zfbc = self.bcco / (1. + 0.5*self.bcco)
#         self.device = param['device']

#         ### the part we want to augment with NN
#         if param['p_prime']:
#             self.p_prime = torch.from_numpy(np.load(param['p_prime'])).type(torch.float64).to(self.device)
#             if self.n_ens > 0:
#                 self.p_prime = self.p_prime.unsqueeze(0)
#         else:
#             self.p_prime = None
#         ### CNN 
#         nb_channels = self.nl * 2  # 2 channels per layer (dq, dp)
#         self.unet = Unet(nb_channels,nb_channels, self.n_ens).to(self.device)
#         self.arr_kwargs = {'dtype':torch.float64, 'device':self.device}

#         # grid
#         self.x, self.y = torch.meshgrid(torch.linspace(0, self.Lx, self.nx, **self.arr_kwargs),
#                                         torch.linspace(0, self.Ly, self.ny, **self.arr_kwargs),
#                                         indexing='ij')
#         self.y0 = 0.5 * self.Ly
#         self.dx = self.Lx / (self.nx-1)
#         self.dy = self.Ly / (self.ny-1)
#         assert self.dx == self.dy, f'dx {self.dx} != dy {self.dy}, must be equal'

#         self.diff_coef = self.a_2 / self.f0**2 / self.dx**4
#         self.hyperdiff_coef = (self.a_4 / self.f0**2) / self.dx**6
#         self.jac_coef =  1. / (self.f0 * self.dx * self.dy)
#         self.bottom_friction_coef = self.delta_ek / (2*np.abs(self.f0)*self.dx**2*(-self.heights[-1])) # (Be)_4 coefficient

#         tau = torch.zeros((self.nx, self.ny, 2), **self.arr_kwargs)
#         tau[:,:,0] = - param['tau0'] * torch.cos(2*torch.pi*(torch.arange(self.ny, **self.arr_kwargs)+0.5)/self.ny).reshape((1, self.ny))
#         self.wind_forcing = (curl_wind(tau, self.dx, self.dy) / (self.f0 * self.heights[0])).unsqueeze(0) # (Be)_1 
#         if self.n_ens > 0:
#             self.wind_forcing = self.wind_forcing.unsqueeze(0)

#         # init matrices
#         self.compute_A_matrix() # constant, vertical stratification
#         self.compute_layer_to_mode_matrices()
#         self.compute_helmoltz_matrices()
#         self.compute_alpha_matrix()
#         self.helmoltz_dst = self.helmoltz_dst.type(torch.float32)

#         # precomputations: not used 
#         #self.beta_y_y0_over_f0 = (self.beta / self.f0) * (self.y - self.y0)

#         # initialize pressure p and potential vorticity q
#         self.p_shape = (self.nl, self.nx, self.ny) if self.n_ens == 0 else (self.n_ens, self.nl, self.nx, self.ny)
#         self.p_shape_flat = self.p_shape[:-2] + (self.nx*self.ny,)
#         self.p = torch.zeros(self.p_shape, **self.arr_kwargs, requires_grad=True).to(self.device)# init to zero
#         self.p_modes = torch.zeros_like(self.p).to(self.device)
#         self.compute_q_over_f0_from_p() # we get q from p using the Helmhotz equation (2)

#         # precompile torch functions
#         self.zfbc = torch.tensor(self.zfbc, **self.arr_kwargs) # convert to Tensor for tracing
#         self.grad_perp = torch.jit.trace(grad_perp, (self.p,)) # orthogonal gradient on staggered (décalée) grid
#         self.inverse_elliptic_dst = torch.jit.trace(inverse_elliptic_dst, (self.q_over_f0[...,1:-1,1:-1], self.helmoltz_dst))
#         self.jacobi_h = torch.jit.trace(jacobi_h, (self.q_over_f0, self.p))
#         self.laplacian_h = torch.jit.trace(laplacian_h, (self.p, self.zfbc))
#         self.laplacian_h_boundaries = torch.jit.trace(laplacian_h_boundaries, (self.p, self.zfbc))
#         self.laplacian_h_nobc = torch.jit.trace(laplacian_h_nobc, (self.p,))
#         self.matmul = torch.jit.trace(matmul, (self.Cl2m, self.q_over_f0, ))


#     def compute_A_matrix(self):
#         A = torch.zeros((self.nl,self.nl), **self.arr_kwargs)
#         A[0,0] = 1./(self.heights[0]*self.reduced_gravities[0])
#         A[0,1] = -1./(self.heights[0]*self.reduced_gravities[0])
#         for i in range(1, self.nl-1):
#             A[i,i-1] = -1./(self.heights[i]*self.reduced_gravities[i-1])
#             A[i,i] = 1./self.heights[i]*(1/self.reduced_gravities[i] + 1/self.reduced_gravities[i-1])
#             A[i,i+1] = -1./(self.heights[i]*self.reduced_gravities[i])
#         A[-1,-1] = 1./(self.heights[self.nl-1]*self.reduced_gravities[self.nl-2])
#         A[-1,-2] = -1./(self.heights[self.nl-1]*self.reduced_gravities[self.nl-2])
#         if self.n_ens > 0:
#             self.A = A.unsqueeze(0)
#         else:
#             self.A = A
#         #self.A = A.unsqueeze(0) if self.n_ens > 0 else A


#     def compute_layer_to_mode_matrices(self):
#         """Matrices to change from layers to modes."""
#         A = self.A[0] if self.n_ens > 0 else self.A
#         lambd_r, R = torch.linalg.eig(A)
#         lambd_l, L = torch.linalg.eig(A.T)
#         self.lambd = lambd_r.real
#         R, L = R.real, L.real
#         self.Cl2m = torch.diag(1./torch.diag(L.T @ R)) @ L.T
#         self.Cm2l = R
#         if self.n_ens > 0:
#             self.Cl2m, self.Cm2l = self.Cl2m.unsqueeze(0), self.Cm2l.unsqueeze(0) # no inplace


#     def compute_helmoltz_matrices(self):
#         self.helmoltz_dst = compute_laplace_dst(self.nx, self.ny, self.dx, self.dy, self.arr_kwargs).reshape((1, self.nx-2, self.ny-2)) / self.f0**2 - self.lambd.reshape((self.nl , 1, 1))
#         constant_field = torch.ones((self.nl, self.nx, self.ny), **self.arr_kwargs) / (self.nx * self.ny)
#         s_solutions = torch.zeros_like(constant_field)
#         s_solutions[:,1:-1,1:-1] = inverse_elliptic_dst(constant_field[:,1:-1,1:-1], self.helmoltz_dst)
#         self.homogeneous_sol = (constant_field +  s_solutions*self.lambd.reshape((self.nl, 1, 1)))[:-1] # ignore last solution correponding to lambd = 0, i.e. Laplace equation
#         if self.n_ens > 0:
#             self.helmoltz_dst, self.homogeneous_sol = self.helmoltz_dst.unsqueeze(0), self.homogeneous_sol.unsqueeze(0)


#     def compute_alpha_matrix(self):
#         (Cm2l, Cl2m, hom_sol) = (self.Cm2l[0], self.Cl2m[0], self.homogeneous_sol[0]) if self.n_ens > 0 else (self.Cm2l, self.Cl2m, self.homogeneous_sol)
#         M = (Cm2l[1:] - Cm2l[:-1])[:self.nl-1,:self.nl-1] * hom_sol.mean((1,2)).reshape((1, self.nl-1))
#         M_inv = torch.linalg.inv(M)
#         alpha_matrix = -M_inv @ (Cm2l[1:,:-1] - Cm2l[:-1,:-1])
#         self.alpha_matrix = alpha_matrix.unsqueeze(0) if self.n_ens > 0 else alpha_matrix


#     def compute_q_over_f0_from_p(self):
#         Ap = (self.A @ self.p.reshape(self.p.shape[:len(self.p.shape)-2]+(-1,))).reshape(self.p.shape)
#         self.q_over_f0 = laplacian_h(self.p, self.zfbc) / (self.f0*self.dx)**2 - Ap + (self.beta / self.f0) * (self.y - self.y0)

#     def compute_u(self):
#         """Compute velocity on staggered grid."""
#         return self.grad_perp(self.p/(self.f0*self.dx))


#     def advection_rhs(self): # RHS = Right Hand Side
#         """Advection diffusion RHS for vorticity, only inside domain""" 
#         # laplacian_h_nobc = only inside the domain
#         rhs = self.jac_coef * self.jacobi_h(self.q_over_f0, self.p)

#         delta2_p = self.laplacian_h(self.p, self.zfbc)
#         if self.a_2 != 0.:
#             rhs = rhs + self.diff_coef * self.laplacian_h_nobc(delta2_p)
#         if self.a_4 != 0.:
#             rhs = rhs - self.hyperdiff_coef * self.laplacian_h_nobc(self.laplacian_h(delta2_p, self.zfbc))

#         #rhs[...,0:1,:,:] += self.wind_forcing # (Be)_1
#         #rhs[...,-1:,:,:] += self.bottom_friction_coef * self.laplacian_h_nobc(self.p[...,-1:,:,:]) # (Be)_4
#         # without inplace operations
#         # Replace the slicing-based in-place operations:
#         rhs_updated_1 = rhs.clone()  # Create a clone to avoid in-place modification

#         # Update rhs without in-place changes:
#         rhs_updated_1[..., 0:1, :, :] = rhs_updated_1[..., 0:1, :, :] + self.wind_forcing
#         rhs_updated_2 = rhs_updated_1.clone()  # Avoid in-place update again

#         rhs_updated_2[..., -1:, :, :] = rhs_updated_2[..., -1:, :, :] + (
#             self.bottom_friction_coef * self.laplacian_h_nobc(self.p[..., -1:, :, :])
#         )

#         rhs = rhs_updated_2  # Reassign the result to `rhs`

#         return rhs


#     def compute_time_derivatives(self):
#         # advect vorticity inside of the domain
#         rhs = self.advection_rhs()
#         #rhs = torch.zeros_like(rhs) # remove this line to use the NN
#         self.dq_over_f0 = F.pad(rhs, (1,1,1,1)) # dq = RHS

#         # Solve helmoltz eq for pressure p inside the domain
#         rhs_helmoltz = self.matmul(self.Cl2m, self.dq_over_f0) # layer to mode
#         dp_modes = F.pad(self.inverse_elliptic_dst(rhs_helmoltz[...,1:-1,1:-1], self.helmoltz_dst), (1,1,1,1))

#         # Ensure mass conservation
#         dalpha =  (self.alpha_matrix @ dp_modes[...,:-1,:,:].mean((-2,-1)).unsqueeze(-1)).unsqueeze(-1)
#         dp_modes[...,:-1,:,:] += dalpha * self.homogeneous_sol # add homogeneous solution
#         # update dp 
#         self.dp = self.matmul(self.Cm2l, dp_modes) # mode to layer

#         # update vorticity q on the boundaries
#         dp_bound = torch.cat([self.dp[...,0,1:-1], self.dp[...,-1,1:-1], self.dp[...,:,0], self.dp[...,:,-1]], dim=-1) # get dp boundaries
#         delta_p_bound = self.laplacian_h_boundaries(self.dp/(self.f0*self.dx)**2, self.zfbc)
#         dq_over_f0_bound = delta_p_bound - self.A @ dp_bound
#         self.dq_over_f0[...,0,1:-1] = dq_over_f0_bound[...,:self.ny-2] # top boundary (no corners)
#         self.dq_over_f0[...,-1,1:-1] = dq_over_f0_bound[...,self.ny-2:2*self.ny-4] # bottom boundary (no corners)
#         self.dq_over_f0[...,0] = dq_over_f0_bound[...,2*self.ny-4:self.nx+2*self.ny-4] # left boundary (corners)
#         self.dq_over_f0[...,-1] = dq_over_f0_bound[...,self.nx+2*self.ny-4:2*self.nx+2*self.ny-4] # right boundary (corners)


#     def format_input_unet(self, x1,x2):
#         if self.n_ens == 0:
#             return rearrange(torch.stack((x1, x2),1), 'nl nc nx ny -> 1 (nl nc) nx ny').type(torch.float32).to(self.device)
#         else:
#             return rearrange(torch.stack((x1, x2),2), 'n nl nc nx ny -> n (nl nc) nx ny').type(torch.float32).to(self.device)
    
#     def step(self):
#         """ Time itegration with Heun (RK2) scheme."""
#         self.compute_time_derivatives() # problème est qu'il le fait inplace 
#         dq_over_f0_0, dp_0 = self.dq_over_f0.clone(), self.dp.clone()
#         #in_unet = self.format_input_unet(self.dq_over_f0, self.dp)
#         #dq_aug, dp_aug = self.unet(in_unet)
#         #dq_over_f0_0 = dq_over_f0_0 + dq_aug
#         #dp_0 = dp_0 + dp_aug
#         self.q_over_f0 = self.q_over_f0 + self.dt * dq_over_f0_0 
#         self.p = self.p + self.dt * dp_0

#         self.compute_time_derivatives()
#         #in_unet = self.format_input_unet(self.dq_over_f0, self.dp)
#         #dq_aug, dp_aug = self.unet(in_unet)
#         self.q_over_f0 = self.q_over_f0 + self.dt * 0.5 * (self.dq_over_f0  - dq_over_f0_0)
#         self.p = self.p + self.dt * 0.5 * (self.dp  - dp_0)


class QGM_differentiable:
    """Implementation of multilayer quasi-geostrophic model
    in variables pressure p and potential vorticity q.
    """

    def __init__(self, param):
        ### parameters of QG equations
        self.nx = param['nx']
        self.Lx = param['Lx']
        self.ny = param['ny']
        self.Ly = param['Ly']
        self.nl = param['nl']
        self.heights = param['heights']
        self.reduced_gravities = param['reduced_gravities']
        self.f0 = param['f0']
        self.a_2 = param['a_2']
        self.a_4 = param['a_4']
        self.beta = param['beta']
        self.delta_ek = param['delta_ek']
        self.dt = param['dt']
        self.bcco = param['bcco']
        self.n_ens = param['n_ens']
        self.zfbc = self.bcco / (1. + 0.5*self.bcco)
        self.device = param['device']

        # ### CNN 
        #nb_channels = self.nl * 2  # 2 channels per layer (dq, dp)
        #self.unet = Unet(nb_channels,nb_channels, self.n_ens).to(self.device)
        self.arr_kwargs = {'dtype':torch.float64, 'device':self.device}

        # grid
        self.x, self.y = torch.meshgrid(torch.linspace(0, self.Lx, self.nx, **self.arr_kwargs),
                                        torch.linspace(0, self.Ly, self.ny, **self.arr_kwargs),
                                        indexing='ij')
        self.x, self.y = self.x.type(torch.float32), self.y.type(torch.float32)
        self.y0 = 0.5 * self.Ly
        self.dx = self.Lx / (self.nx-1)
        self.dy = self.Ly / (self.ny-1)
        assert self.dx == self.dy, f'dx {self.dx} != dy {self.dy}, must be equal'

        self.diff_coef = self.a_2 / self.f0**2 / self.dx**4
        self.hyperdiff_coef = (self.a_4 / self.f0**2) / self.dx**6
        self.jac_coef =  1. / (self.f0 * self.dx * self.dy)
        self.bottom_friction_coef = self.delta_ek / (2*np.abs(self.f0)*self.dx**2*(-self.heights[-1])) # (Be)_4 coefficient

        tau = torch.zeros((self.nx, self.ny, 2), **self.arr_kwargs)
        tau[:,:,0] = - param['tau0'] * torch.cos(2*torch.pi*(torch.arange(self.ny, **self.arr_kwargs)+0.5)/self.ny).reshape((1, self.ny))
        self.wind_forcing = (curl_wind(tau, self.dx, self.dy) / (self.f0 * self.heights[0])).unsqueeze(0) # (Be)_1 
        if self.n_ens > 0:
            self.wind_forcing = self.wind_forcing.unsqueeze(0)

        # init matrices
        self.A = self.compute_A_matrix() # constant, vertical stratification
        self.Cl2m, self.Cm2l, self.lambd = self.compute_layer_to_mode_matrices()
        self.helmoltz_dst, self.homogeneous_sol = self.compute_helmoltz_matrices()
        self.alpha_matrix = self.compute_alpha_matrix()
        self.helmoltz_dst = self.helmoltz_dst.type(torch.float32)

        # put to right type
        self.A = self.A.type(torch.float32)
        self.Cl2m = self.Cl2m.type(torch.float32)
        self.Cm2l = self.Cm2l.type(torch.float32)
        self.lambd = self.lambd.type(torch.float32)
        self.alpha_matrix = self.alpha_matrix.type(torch.float32)

        # initialize pressure p and potential vorticity q
        self.p_shape = (self.nl, self.nx, self.ny) if self.n_ens == 0 else (self.n_ens, self.nl, self.nx, self.ny)
        self.p0 = torch.zeros(self.p_shape, **self.arr_kwargs).type(torch.float32)
        q_over_f0 = self.compute_q_over_f0_from_p(self.p0) # we get q from p using the Helmhotz equation (2)

        # precompile torch functions
        self.zfbc = torch.tensor(self.zfbc, **self.arr_kwargs) # convert to Tensor for tracing
        self.grad_perp = torch.jit.trace(grad_perp, (self.p0,)) # orthogonal gradient on staggered (décalée) grid
        self.inverse_elliptic_dst = torch.jit.trace(inverse_elliptic_dst, (q_over_f0[...,1:-1,1:-1], self.helmoltz_dst))
        self.jacobi_h = torch.jit.trace(jacobi_h, (q_over_f0, self.p0))
        self.laplacian_h = torch.jit.trace(laplacian_h, (self.p0, self.zfbc))
        self.laplacian_h_boundaries = torch.jit.trace(laplacian_h_boundaries, (self.p0, self.zfbc))
        self.laplacian_h_nobc = torch.jit.trace(laplacian_h_nobc, (self.p0,))
        self.matmul = torch.jit.trace(matmul, (self.Cl2m, q_over_f0, ))

    ### INITIALIZATION METHODS
    def compute_A_matrix(self):
        A = torch.zeros((self.nl,self.nl), **self.arr_kwargs)
        A[0,0] = 1./(self.heights[0]*self.reduced_gravities[0])
        A[0,1] = -1./(self.heights[0]*self.reduced_gravities[0])
        for i in range(1, self.nl-1):
            A[i,i-1] = -1./(self.heights[i]*self.reduced_gravities[i-1])
            A[i,i] = 1./self.heights[i]*(1/self.reduced_gravities[i] + 1/self.reduced_gravities[i-1])
            A[i,i+1] = -1./(self.heights[i]*self.reduced_gravities[i])
        A[-1,-1] = 1./(self.heights[self.nl-1]*self.reduced_gravities[self.nl-2])
        A[-1,-2] = -1./(self.heights[self.nl-1]*self.reduced_gravities[self.nl-2])
        if self.n_ens > 0:
            return A.unsqueeze(0)
        else:
            return A


    def compute_layer_to_mode_matrices(self):
        """Matrices to change from layers to modes."""
        A = self.A[0] if self.n_ens > 0 else self.A
        lambd_r, R = torch.linalg.eig(A)
        lambd_l, L = torch.linalg.eig(A.T)
        lambd = lambd_r.real
        R, L = R.real, L.real
        Cl2m = torch.diag(1./torch.diag(L.T @ R)) @ L.T
        Cm2l = R
        if self.n_ens > 0:
            return Cl2m.unsqueeze(0), Cm2l.unsqueeze(0), lambd
        else:
            return Cl2m, Cm2l, lambd


    def compute_helmoltz_matrices(self):
        helmoltz_dst = compute_laplace_dst(self.nx, self.ny, self.dx, self.dy, self.arr_kwargs).reshape((1, self.nx-2, self.ny-2)) / self.f0**2 - self.lambd.reshape((self.nl , 1, 1))
        constant_field = torch.ones((self.nl, self.nx, self.ny), **self.arr_kwargs) / (self.nx * self.ny)
        s_solutions = torch.zeros_like(constant_field)
        s_solutions[:,1:-1,1:-1] = inverse_elliptic_dst(constant_field[:,1:-1,1:-1], helmoltz_dst)
        homogeneous_sol = (constant_field +  s_solutions*self.lambd.reshape((self.nl, 1, 1)))[:-1] # ignore last solution correponding to lambd = 0, i.e. Laplace equation
        if self.n_ens > 0:
            return helmoltz_dst.unsqueeze(0), homogeneous_sol.unsqueeze(0) # no inplace
        else:
            return helmoltz_dst, homogeneous_sol
        

    def compute_alpha_matrix(self):
        (Cm2l, Cl2m, hom_sol) = (self.Cm2l[0], self.Cl2m[0], self.homogeneous_sol[0]) if self.n_ens > 0 else (self.Cm2l, self.Cl2m, self.homogeneous_sol)
        M = (Cm2l[1:] - Cm2l[:-1])[:self.nl-1,:self.nl-1] * hom_sol.mean((1,2)).reshape((1, self.nl-1))
        M_inv = torch.linalg.inv(M)
        alpha_matrix = -M_inv @ (Cm2l[1:,:-1] - Cm2l[:-1,:-1])
        if self.n_ens > 0:
            return alpha_matrix.unsqueeze(0)
        else:
            return alpha_matrix

    ### COMPUTATION METHODS
    def compute_q_over_f0_from_p(self, p):
        Ap = (self.A @ p.reshape(p.shape[:len(p.shape)-2]+(-1,))).reshape(p.shape)
        q_over_f0 = laplacian_h(p, self.zfbc) / (self.f0*self.dx)**2 - Ap + (self.beta / self.f0) * (self.y - self.y0)
        return q_over_f0

    def compute_u(self, p):
        """Compute velocity on staggered grid."""
        return self.grad_perp(p/(self.f0*self.dx))

    def advection_rhs(self, q_over_f0, p): # RHS = Right Hand Side
        """Advection diffusion RHS for vorticity, only inside domain""" 
        # laplacian_h_nobc = only inside the domain
        rhs = self.jac_coef * self.jacobi_h(q_over_f0, p)

        delta2_p = self.laplacian_h(p, self.zfbc)
        if self.a_2 != 0.:
            rhs = rhs + self.diff_coef * self.laplacian_h_nobc(delta2_p)
        if self.a_4 != 0.:
            rhs = rhs - self.hyperdiff_coef * self.laplacian_h_nobc(self.laplacian_h(delta2_p, self.zfbc))

        rhs[...,0:1,:,:] += self.wind_forcing # (Be)_1
        rhs[...,-1:,:,:] += self.bottom_friction_coef * self.laplacian_h_nobc(p[...,-1:,:,:]) # (Be)_4

        return rhs

    def compute_time_derivatives(self, y0):
        q_over_f0, p = y0[0], y0[1]
        # advect vorticity inside of the domain
        rhs = self.advection_rhs(q_over_f0, p)
        #rhs = torch.zeros_like(rhs) # remove this line to use the NN
        dq_over_f0 = F.pad(rhs, (1,1,1,1)) # dq = RHS

        # Solve helmoltz eq for pressure p inside the domain
        rhs_helmoltz = self.matmul(self.Cl2m, dq_over_f0) # layer to mode
        dp_modes = F.pad(self.inverse_elliptic_dst(rhs_helmoltz[...,1:-1,1:-1], self.helmoltz_dst), (1,1,1,1))

        # Ensure mass conservation
        dalpha =  (self.alpha_matrix @ dp_modes[...,:-1,:,:].mean((-2,-1)).unsqueeze(-1)).unsqueeze(-1)
        dp_modes[...,:-1,:,:] += dalpha * self.homogeneous_sol # add homogeneous solution
        # update dp 
        dp = self.matmul(self.Cm2l, dp_modes) # mode to layer

        # update vorticity q on the boundaries
        dp_bound = torch.cat([dp[...,0,1:-1], dp[...,-1,1:-1], dp[...,:,0], dp[...,:,-1]], dim=-1) # get dp boundaries
        delta_p_bound = self.laplacian_h_boundaries(dp/(self.f0*self.dx)**2, self.zfbc)
        dq_over_f0_bound = delta_p_bound - self.A @ dp_bound
        dq_over_f0[...,0,1:-1] = dq_over_f0_bound[...,:self.ny-2] # top boundary (no corners)
        dq_over_f0[...,-1,1:-1] = dq_over_f0_bound[...,self.ny-2:2*self.ny-4] # bottom boundary (no corners)
        dq_over_f0[...,0] = dq_over_f0_bound[...,2*self.ny-4:self.nx+2*self.ny-4] # left boundary (corners)
        dq_over_f0[...,-1] = dq_over_f0_bound[...,self.nx+2*self.ny-4:2*self.nx+2*self.ny-4] # right boundary (corners)
        return dq_over_f0, dp


    # def format_input_unet(self, x1,x2):
    #     if self.n_ens == 0:
    #         return rearrange(torch.stack((x1, x2),1), 'nl nc nx ny -> 1 (nl nc) nx ny').type(torch.float32).to(self.device)
    #     else:
    #         return rearrange(torch.stack((x1, x2),2), 'n nl nc nx ny -> n (nl nc) nx ny').type(torch.float32).to(self.device)
    
    # def step(self, q_over_f0, p):
    #     """ Time itegration with Heun (RK2) scheme."""
    #     dq_over_f0, dp = self.compute_time_derivatives(q_over_f0, p) 
    #     # augmented dynamics   
    #     in_unet = self.format_input_unet(dq_over_f0, dp)
    #     dq_aug, dp_aug = self.unet(in_unet)
    #     dq_over_f0 = dq_over_f0 + dq_aug # f(y(i))
    #     dp = dp + dp_aug # f(y(i))
    #     # time step 1 with Heun (RK2)
    #     # y1(i+1) = y(i) + dt * f(y(i))
    #     q_over_f0_1 = q_over_f0 + self.dt * dq_over_f0
    #     p_1 = p + self.dt * dp

    #     dq_over_f0_2, dp_2 = self.compute_time_derivatives(q_over_f0_1, p_1) 
    #     # augmented dynamics
    #     in_unet = self.format_input_unet(dq_over_f0_2, dp_2)
    #     dq_aug, dp_aug = self.unet(in_unet)
    #     dq_over_f0_2 = dq_over_f0_2 + dq_aug # f(y1(i+1))
    #     dp_2 = dp_2 + dp_aug # f(y1(i+1))
    #     # time step 2 with Heun (RK2)
    #     # y2(i+1) = y(i) + dt * 0.5 * (f(y(i)) + f(y1(i+1)))
    #     new_q_over_f0 = q_over_f0 + self.dt * 0.5 * (dq_over_f0 + dq_over_f0_2)
    #     new_p = p + self.dt * 0.5 * (dp + dp_2)
    #     return new_q_over_f0, new_p
    #     #return q_over_f0_1, p_1
    

class HybridForecaster(nn.Module):
    """Combines a physical model with a neural network for forecasting"""
    model_phy: QGM_differentiable
    model_aug: nn.Module
    dt: float 
    num_steps: int
    integration_method: str 

    def __init__(self, model_phy, model_aug, dt, num_steps, integration_method, is_augmented):
        super().__init__()
        self.model_phy = model_phy
        self.model_aug = model_aug
        self.dt = dt
        self.num_steps = num_steps
        self.integration_method = integration_method
        self.int_ = odeint
        self.is_augmented = is_augmented

    def forward(self, y0, t):
        res = self.int_(self.derivative_estimator, y0, t, method=self.integration_method) 
        #self.int_(self.derivative_estimator, y0, self.dt, self.num_steps, RK_tableaux[self.integration_method])
        return res
    
    def derivative_estimator(self, t, y0):
        dq_over_f0, dp = self.model_phy.compute_time_derivatives(y0)
        if self.is_augmented:
            # augmented dynamics
            in_unet = rearrange(y0, 'nc nl nx ny -> 1 (nc nl) nx ny') #.type(torch.float32)
            dq_aug, dp_aug = self.model_aug(in_unet)
            dq_over_f0 = dq_over_f0 + dq_aug
            dp = dp + dp_aug
        return torch.stack((dq_over_f0, dp),0)


if __name__ == "__main__":
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

    qg_multilayer = QGM_differentiable(param)
    nb_channels = param["nl"] * 2  # 2 channels per layer (dq, dp)
    unet = Unet(nb_channels, nb_channels, param["n_ens"]).to(param["device"])
    net = HybridForecaster(
        model_phy=qg_multilayer,
        model_aug=unet,
        dt=param["dt"],
        num_steps=10,
        integration_method="heun2",
        is_augmented=True,
    )
    #print(net)
    # init
    p = qg_multilayer.p0
    q_over_f0 = qg_multilayer.compute_q_over_f0_from_p(p)
    y0 = torch.stack((q_over_f0,p),0).to(param["device"]) # nc nl nx ny
    t = torch.arange(0, 10, 1).type(torch.float32).to(param["device"]) # time steps
    #print(net.derivative_estimator(y0).shape)
    out = net(y0, t)
    print(out.shape)
