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

        #normals wijzen van buiten naar object
        self.normals = []
        
        for i in range(1, pixels[0]-1):
            for j in range(1, pixels[1] - 1):
                if not object_mask[i, j]:
                    # Check neighbors to find boundary sides
                    if object_mask[i-1, j]: self.normals.append((i, j, -1, 0))   # left
                    if object_mask[i+1, j]: self.normals.append((i, j, +1, 0))   # right
                    if object_mask[i, j-1]: self.normals.append((i, j, 0, -1))   # down
                    if object_mask[i, j+1]: self.normals.append((i, j, 0, +1))   # up
        
        self.mu = 1.8e-5        # Dynamic viscosity
        self.Pr = 0.72          # Prandtl number for air
        self.kappa = self.mu * self.gamma / self.Pr

        self.time = 0

        self.rho = np.ones(pixels) * rho0
        self.u = np.ones(pixels) * v0[0]
        self.v = np.ones(pixels) * v0[1]
        self.uv = np.sqrt((np.square(self.u) + np.square(self.v)))
        self.e = np.ones(pixels) * e0  # interne energie
        self.p = np.multiply(self.rho, self.e) * (self.gamma - 1)

    @property
    def dxy(self):
        return self.size[0] / self.pixels[0], self.size[1] / self.pixels[1]
    
    def deuler(self, vars):
        #implementatie euler vergelijkingen.
        #hier eventueel veranderen naar navier-stokes.
        rho, u, v, e = vars[0], vars[1], vars[2], vars[3]
        p = np.multiply(rho, e) * (self.gamma - 1)

        drho_dx, drho_dy = ddxy(rho, self.dxy)
        du_dx, du_dy = ddxy(u, self.dxy)
        dv_dx, dv_dy = ddxy(v, self.dxy)
        dp_dx, dp_dy = ddxy(p, self.dxy)
        de_dx, de_dy = ddxy(e, self.dxy)

        div_u = du_dx + dv_dy

        Fx, Fy = rho * 0, rho * 0

        drho_dt = -(self.u * drho_dx + v * drho_dy + rho * div_u)
        du_dt = -(self.u * du_dx + v * du_dy + dp_dx / rho - Fx)
        dv_dt = -(self.u * dv_dx + v * dv_dy + dp_dy / rho - Fy)
        de_dt = -(self.u * de_dx + v * de_dy + p / rho * div_u)

        return np.array([drho_dt, du_dt, dv_dt, de_dt])

    def d(self, vars):
        # Retrieve current state variables
        rho, u, v, e = vars[0], vars[1], vars[2], vars[3]
        p = np.multiply(rho, e) * (self.gamma - 1)

        # --- 1. Calculate First Derivatives (Same as Euler) ---
        drho_dx, drho_dy = ddxy(rho, self.dxy)
        du_dx, du_dy = ddxy(u, self.dxy)
        dv_dx, dv_dy = ddxy(v, self.dxy)
        dp_dx, dp_dy = ddxy(p, self.dxy)
        de_dx, de_dy = ddxy(e, self.dxy)

        div_u = du_dx + dv_dy

        # --- 2. Calculate Second Derivatives
        laplace_u = laplacian(u, self.dxy)
        laplace_v = laplacian(v, self.dxy)
        laplace_e = laplacian(e, self.dxy)
        
        ddiv_u_dx, ddiv_u_dy = ddxy(div_u, self.dxy)

        # --- 3. Calculate Viscous Terms (V_u, V_v, V_e) ---
        # Simplified viscous terms for momentum (using Stokes' hypothesis, lambda = -2/3*mu)
        viscous_u_force = (self.mu / rho) * (laplace_u + (1/3) * ddiv_u_dx)
        viscous_v_force = (self.mu / rho) * (laplace_v + (1/3) * ddiv_u_dy)
        
        # Viscous term for energy (heat conduction)
        viscous_e_term = (self.kappa / rho) * laplace_e
        
        # --- 4. Time Derivatives (Euler terms + Viscous terms) ---
        Fx, Fy = rho * 0, rho * 0 # Fx and Fy are your body forces (e.g., acceleration)

        drho_dt = -(self.u * drho_dx + v * drho_dy + rho * div_u)   
        
        # Momentum (Euler + Viscous Force)
        # The term is added because the viscous stress is a *force* acting on the fluid.
        du_dt = -(self.u * du_dx + v * du_dy + dp_dx / rho - Fx) + viscous_u_force
        dv_dt = -(self.u * dv_dx + v * dv_dy + dp_dy / rho - Fy) + viscous_v_force
        
        # Energy (Euler + Heat Conduction)
        de_dt = -(self.u * de_dx + v * de_dy + p / rho * div_u) + viscous_e_term
    
        return np.array([drho_dt, du_dt, dv_dt, de_dt])

    def rk4(self, u, h):
        # moet beter zijn dan direct euler
        k1 = self.d(u)
        k2 = self.d(u + h*k1/2)
        k3 = self.d(u + h*k2/2)
        k4 = self.d(u + h*k3)
        return u + h/6*(k1 + 2*k2 + 2+k3 + k4)

    def step(self, dt):
        variables = np.array([self.rho, self.u, self.v, self.e])

        self.p = np.multiply(self.rho, self.e) * (self.gamma - 1)

        variables = self.rk4(variables, dt)
        rho_new, u_new, v_new, e_new = variables[0], variables[1], variables[2], variables[3]

        self.uv = np.sqrt((np.square(u_new) + np.square(v_new)))    
        self.c = np.sqrt(np.clip(self.gamma*self.p/np.clip(rho_new, a_min=0.1, a_max=100), a_min=0, a_max=1000_000_000))

        nu = 0.05 * self.dxy[0] * (self.uv + self.c)
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
        
        self.drag, self.lift = 0, 0
        for (i, j, dx, dy) in self.normals:
            self.drag += self.p[i][j] * dx * self.dxy[0]
            self.lift += self.p[i][j] * dy * self.dxy[1]