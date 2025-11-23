import numpy as np

def ddx(f, dx):
    d = np.zeros_like(f)
    d[..., 1:-1, :] = (f[..., 2:, :] - f[..., :-2, :]) / (2 * dx)
    d[..., 0, :] = (f[..., 1, :] - f[..., 0, :]) / dx      # forward diff at left boundary
    d[..., -1, :] = (f[..., -1, :] - f[..., -2, :]) / dx    # backward diff at right boundary
    return d

def ddy(f, dy):
    d = np.zeros_like(f)
    d[..., :, 1:-1] = (f[..., :, 2:] - f[..., :, :-2]) / (2 * dy)
    d[..., :, 0] = (f[..., :, 1] - f[..., :, 0]) / dy       # forward diff at bottom boundary
    d[..., :, -1] = (f[..., :, -1] - f[..., :, -2]) / dy    # backward diff at top boundary
    return d

def ddxy(f, dxy):
    return ddx(f, dxy[0]), ddy(f, dxy[1])

def laplacian(f, dxy):
    grad_x, grad_y = ddx(f, dxy[0]), ddy(f, dxy[1])
    return ddx(grad_x, dxy[0]) + ddy(grad_y, dxy[1])
