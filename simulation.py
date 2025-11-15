import numpy as np
from derivatives import *

class Simluation:
    def __init__(self, size, pixels, gamma, rho0, e0, v0, object_mask):
        self.size = size
        self.pixels = pixels
        self.gamma = gamma
        self.rho0 = rho0
        self.e0 = e0
        self.object_mask = object_mask

        self.time = 0

        self.rho = np.ones(pixels) * rho0
        self.u = np.ones(pixels) * v0[0]
        self.v = np.ones(pixels) * v0[1]
        self.e = np.ones(pixels) * e0  # interne energie
        self.p = np.multiply(self.rho, self.e) * (self.gamma - 1)

    @property
    def dxy(self):
        return self.size[0] / self.pixels[0], self.size[1] / self.pixels[1]


    def step(self, dt):
        self.p = np.multiply(self.rho, self.e) * (self.gamma - 1)
        drho_dx, drho_dy = ddxy(self.rho, self.dxy)
        du_dx, du_dy = ddxy(self.u, self.dxy)
        dv_dx, dv_dy = ddxy(self.v, self.dxy)
        dp_dx, dp_dy = ddxy(self.p, self.dxy)
        de_dx, de_dy = ddxy(self.e, self.dxy)

        div_u = du_dx + dv_dy

        Fx, Fy = self.rho0 * 0, self.rho0 * 0

        # Tijdstappen (expliciet Euler)
        rho_new = self.rho - dt * (self.u * drho_dx + self.v * drho_dy + self.rho * div_u)
        u_new = self.u - dt * (self.u * du_dx + self.v * du_dy + dp_dx / self.rho - Fx)
        v_new = self.v - dt * (self.u * dv_dx + self.v * dv_dy + dp_dy / self.rho - Fy)
        e_new = self.e - dt * (self.u * de_dx + self.v * de_dy + self.p / self.rho * div_u)
        
        self.uv = np.sqrt((np.square(u_new) + np.square(v_new)))
        self.c = np.sqrt(np.clip(self.gamma*self.p/np.clip(rho_new, a_min=0.1, a_max=100), a_min=0, a_max=1000_000_000))

        nu = 0.01 * self.dxy[0] * (self.uv + self.c)
        for m in [rho_new, u_new, v_new, e_new]:
            # Calculate Laplacian (diffusion) using slicing
            laplacian = np.zeros_like(m)
            
            # Interior points (standard 5-point stencil)
            laplacian[1:-1, 1:-1] = (m[2:, 1:-1] + m[:-2, 1:-1] + m[1:-1, 2:] + m[1:-1, :-2] - 4 * m[1:-1, 1:-1])

            # Edges (Need to handle the boundaries carefully, use the nearest interior point)
            # Left/Right Edges (x-direction, uses the outflow B.C. where f[0]=f[1] and f[-1]=f[-2])
            laplacian[0, 1:-1] = (m[1, 1:-1] + m[1, 1:-1] + m[0, 2:] + m[0, :-2] - 4 * m[0, 1:-1]) # f[i-1] is effectively f[1]
            laplacian[-1, 1:-1] = (m[-2, 1:-1] + m[-2, 1:-1] + m[-1, 2:] + m[-1, :-2] - 4 * m[-1, 1:-1]) # f[i+1] is effectively f[-2]
            
            # Bottom/Top Edges (y-direction) - simpler to handle via the main B.C. loop above
            # The main outflow B.C.s m[0,:]=m[1,:] etc. handle the corners and edges already.
            laplacian[1:-1, 0] = (m[1:-1, 1] + m[1:-1, 1] + m[2:, 0] + m[:-2, 0] - 4 * m[1:-1, 0])
            laplacian[1:-1, -1] = (m[1:-1, -2] + m[1:-1, -2] + m[2:, -1] + m[:-2, -1] - 4 * m[1:-1, -1])
            
            # This implementation simplifies the boundaries by assuming the outflow B.C. is sufficient
            # and that the diffusion on the edges is small enough to ignore or is handled by the B.C. loop.
            
            # A safer, more robust approach is to just ensure the corners/edges are not used or are handled:
            laplacian[:, 0] = laplacian[:, 1] # Outflow for the diffusion itself
            laplacian[:, -1] = laplacian[:, -2]
            
            # Update with Diffusion
            m += nu * laplacian

        for m in [rho_new, u_new, v_new, e_new]:
            m[0, :] = m[1, :]
            m[-1, :] = m[-2, :]
            m[:, 0] = m[:, 1]
            m[:, -1] = m[:, -2]

        self.rho, self.u, self.v, self.e = rho_new, u_new, v_new, e_new
        self.rho[self.object_mask] = self.rho0
        self.u[self.object_mask] = 0.0
        self.v[self.object_mask] = 0.0
        self.e[self.object_mask] = self.e0
