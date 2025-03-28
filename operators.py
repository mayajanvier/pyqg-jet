import numpy as np
import torch
import torch.nn.functional as F


## functions to solve elliptic equation with homogeneous boundary conditions
def compute_laplace_dst(nx, ny, dx, dy, arr_kwargs):
    """Discrete sine transform of the 2D centered discrete laplacian
    operator."""
    x, y = torch.meshgrid(torch.arange(1,nx-1, **arr_kwargs),
                          torch.arange(1,ny-1, **arr_kwargs),
                          indexing='ij')
    return 2*(torch.cos(torch.pi/(nx-1)*x) - 1)/dx**2 + 2*(torch.cos(torch.pi/(ny-1)*y) - 1)/dy**2


def dstI1D(x, norm='ortho'):
    """1D type-I discrete sine transform."""
    return torch.fft.irfft(-1j*F.pad(x, (1,1)), dim=-1, norm=norm)[...,1:x.shape[-1]+1]


def dstI2D(x, norm='ortho'):
    """2D type-I discrete sine transform."""
    return dstI1D(dstI1D(x, norm=norm).transpose(-1,-2), norm=norm).transpose(-1,-2)


def inverse_elliptic_dst(f, operator_dst):
    """Inverse elliptic operator (e.g. Laplace, Helmoltz)
       using float32 discrete sine transform."""
    # return dstI2D((dstI2D(f.type(torch.float32)) / operator_dst).type(torch.float64))
    return dstI2D(dstI2D(f.type(torch.float32)) / operator_dst).type(torch.float64)


## discrete spatial differential operators
def jacobi_h(f, g):
    """Arakawa discretisation of Jacobian J(f,g).
       Scalar fields f and g must have the same dimension.
       Grid is regular and dx = dy."""
    dx_f = f[...,2:,:] - f[...,:-2,:]
    dx_g = g[...,2:,:] - g[...,:-2,:]
    dy_f = f[...,2:] - f[...,:-2]
    dy_g = g[...,2:] - g[...,:-2]
    return (
            (   dx_f[...,1:-1] * dy_g[...,1:-1,:] - dx_g[...,1:-1] * dy_f[...,1:-1,:]  ) +
            (   (f[...,2:,1:-1] * dy_g[...,2:,:] - f[...,:-2,1:-1] * dy_g[...,:-2,:]) -
                (f[...,1:-1,2:]  * dx_g[...,2:] - f[...,1:-1,:-2] * dx_g[...,:-2])     ) +
            (   (g[...,1:-1,2:] * dx_f[...,2:] - g[...,1:-1,:-2] * dx_f[...,:-2]) -
                (g[...,2:,1:-1] * dy_f[...,2:,:] - g[...,:-2,1:-1] * dy_f[...,:-2,:])  )
           ) / 12.


def laplacian_h_boundaries(f, fc): # boundaries
    return fc*(torch.cat([f[...,1,1:-1],f[...,-2,1:-1], f[...,1], f[...,-2]], dim=-1) -
               torch.cat([f[...,0,1:-1],f[...,-1,1:-1], f[...,0], f[...,-1]], dim=-1))


def laplacian_h_nobc(f): # inside the domain
    return (f[...,2:,1:-1] + f[...,:-2,1:-1] + f[...,1:-1,2:] + f[...,1:-1,:-2]
            - 4*f[...,1:-1,1:-1])


def matmul(M, f):
    return (M @ f.reshape(f.shape[:-2] + (-1,))).reshape(f.shape)


def laplacian_h(f, fc):
    # laplacian_h = laplacian_h_nobc(f) + laplacian_h_boundaries(f, fc)
    delta_f = torch.zeros_like(f)
    delta_f[...,1:-1,1:-1] = laplacian_h_nobc(f)
    delta_f_bound = laplacian_h_boundaries(f, fc)
    nx, ny = f.shape[-2:]
    delta_f[...,0,1:-1] = delta_f_bound[...,:ny-2]
    delta_f[...,-1,1:-1] = delta_f_bound[...,ny-2:2*ny-4]
    delta_f[...,0] = delta_f_bound[...,2*ny-4:nx+2*ny-4]
    delta_f[...,-1] = delta_f_bound[...,nx+2*ny-4:2*nx+2*ny-4]
    return delta_f


def grad_perp(f):
    """Orthogonal gradient computed ...,on staggered grid."""
    return f[...,:-1] - f[...,1:], f[...,1:,:] - f[...,:-1,:]


def curl_wind(tau, dx, dy):
    tau_x = 0.5 * (tau[:-1,:,0] + tau[1:,:,0])
    tau_y = 0.5 * (tau[:,:-1,1] + tau[:,1:,1])
    curl_stagg = (tau_y[1:] - tau_y[:-1]) / dx - (tau_x[:,1:] - tau_x[:,:-1]) / dy
    return  0.25*(curl_stagg[:-1,:-1] + curl_stagg[:-1,1:] + curl_stagg[1:,:-1] + curl_stagg[1:,1:])