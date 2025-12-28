import numpy as np

# Afgelijde van waarde naar x-coordinaat
def ddx(f, dx):
    d = np.zeros_like(f)
    d[..., 1:-1, :] = (f[..., 2:, :] - f[..., :-2, :]) / (2 * dx)
    # Waarden aan de rand
    d[..., 0, :] = (f[..., 1, :] - f[..., 0, :]) / dx
    d[..., -1, :] = (f[..., -1, :] - f[..., -2, :]) / dx
    return d

# Afgelijde van waarde naar y-coordinaat
def ddy(f, dy):
    d = np.zeros_like(f)
    d[..., :, 1:-1] = (f[..., :, 2:] - f[..., :, :-2]) / (2 * dy)
    # Waarden aan de rand
    d[..., :, 0] = (f[..., :, 1] - f[..., :, 0]) / dy
    d[..., :, -1] = (f[..., :, -1] - f[..., :, -2]) / dy 
    return d

# tuple met beide afgeleiden
def ddxy(f, dxy):
    return ddx(f, dxy[0]), ddy(f, dxy[1])

# Laplacian van het veld f (hoe groot de waarden rond elk punt zijn vergeleken met dat punt zelf)
def laplacian(f, dxy):
    grad_x, grad_y = ddx(f, dxy[0]), ddy(f, dxy[1])
    return ddx(grad_x, dxy[0]) + ddy(grad_y, dxy[1])