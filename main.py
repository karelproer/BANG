import numpy as np
import matplotlib.pyplot as plt

# Parameters
Nx, Ny = 200,200
Lx, Ly = 1.0, 1.0
ax, ay = 0, 0.0
dx, dy = Lx / Nx, Ly / Ny
dt = 0.00000001 # in seconden
Nt = 400000
gamma = 1.4
rho0, e0 = 1, 214_000

# Rooster
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y, indexing='ij')

# Beginwaarden
rho = np.ones((Nx, Ny)) * rho0
u = np.ones((Nx, Ny)) * 30
v = np.ones((Nx, Ny)) * 0
e = np.ones((Nx, Ny)) * e0  # interne energie

Object_Mask = np.zeros((Nx, Ny), dtype=bool)

# Find the indices corresponding to the object's boundaries
# Note: X and Y are already defined as meshgrids from 0 to 1
#is_inside_x = (X >= 0.4) & (X <= 0.6)
#is_inside_y = (Y >= 0.4) & (Y <= 0.6)
#Object_Mask = is_inside_x & is_inside_y

M = 0.06 #Maximaal camber %
P = 0.396 #max camber position in %
T = 0.12 #max thickness
a0, a1, a2, a3, a4 = 0.2969, -0.126, -0.3156, 0.2843, -0.1036 #standaard waarden

# Scale + position wing
x_wing = (X - 0.3) / 0.5   # move/scale in x
y_wing = (Y - 0.5) / 0.5   # move/scale in y

# make safe copy to avoid sqrt issues at x=0
xch = np.clip(x_wing, 0.0, 1.0)

# camber yc (piecewise) - vectorized
yc = np.zeros_like(xch)
left = xch <= P
right = ~left
yc[left]  = (M / (P**2)) * (2*P*xch[left] - xch[left]**2)
yc[right] = (M / ((1-P)**2)) * ((1 - 2*P) + 2*P*xch[right] - xch[right]**2)

# derivative dyc/dx (piecewise)
dyc_dx = np.zeros_like(xch)
dyc_dx[left]  = (2*M / (P**2)) * (P - xch[left])
dyc_dx[right] = (2*M / ((1-P)**2)) * (P - xch[right])

# thickness distribution 
yt = (T / 0.2) * (
    a0 * np.sqrt(xch) +
    a1 * xch +
    a2 * xch**2 +
    a3 * xch**3 +
    a4 * xch**4
)

# normal angle to camber line
theta = np.arctan(dyc_dx)

# upper and lower surface coordinates (in wing-local coords)
y_upper = yc + yt * np.cos(theta)
y_lower = yc - yt * np.cos(theta)
x_upper = xch - yt * np.sin(theta)
x_lower = xch + yt * np.sin(theta)

Object_Mask = (x_lower >= 0) & (x_upper <= 1) & (y_wing >= y_lower) & (y_wing <= y_upper)

def ddx(f):
    d = np.zeros_like(f)
    d[1:-1, :] = (f[2:, :] - f[:-2, :]) / (2 * dx)
    d[0, :] = (f[1, :] - f[0, :]) / dx      # forward diff at left boundary
    d[-1, :] = (f[-1, :] - f[-2, :]) / dx    # backward diff at right boundary
    return d

def ddy(f):
    d = np.zeros_like(f)
    d[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2 * dy)
    d[:, 0] = (f[:, 1] - f[:, 0]) / dy       # forward diff at bottom boundary
    d[:, -1] = (f[:, -1] - f[:, -2]) / dy    # backward diff at top boundary
    return d


# Visualisatie setup
plt.ion()
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5))
im1 = ax1.imshow(rho.T, origin='lower', extent=[0, Lx, 0, Ly], cmap='inferno', vmin=0.9999, vmax=1.0001)
ax1.set_title("Density")
plt.colorbar(im1, ax=ax1)

im2 = ax2.imshow(e.T, origin='lower', extent=[0, Lx, 0, Ly], cmap='inferno')
ax2.set_title("Internal Energy")
plt.colorbar(im2, ax=ax2)

im3 = ax3.imshow(u.T, origin='lower', extent=[0, Lx, 0, Ly], cmap='inferno', vmin=0, vmax=1)
ax3.set_title("Speed")
plt.colorbar(im3, ax=ax3)

for n in range(Nt):
    p = np.multiply(rho, e) * (gamma - 1)

    # Afgeleiden
    drho_dx, drho_dy = ddx(rho), ddy(rho)
    du_dx, du_dy = ddx(u), ddy(u)
    dv_dx, dv_dy = ddx(v), ddy(v)
    dp_dx, dp_dy = ddx(p), ddy(p)
    de_dx, de_dy = ddx(e), ddy(e)

    div_u = du_dx + dv_dy

    Fx, Fy = rho * ax, rho * ay

    # Tijdstappen (expliciet Euler)
    rho_new = rho - dt * (u * drho_dx + v * drho_dy + rho * div_u)
    u_new = u - dt * (u * du_dx + v * du_dy + dp_dx / rho - Fx)
    v_new = v - dt * (u * dv_dx + v * dv_dy + dp_dy / rho - Fy)
    e_new = e - dt * (u * de_dx + v * de_dy + p / rho * div_u)

    for m in [rho_new, u_new, v_new, e_new]:
        m[0, :] = m[1, :]
        m[-1, :] = m[-2, :]
        m[:, 0] = m[:, 1]
        m[:, -1] = m[:, -2]

    nu = 0.0001
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

    rho, u, v, e = rho_new, u_new, v_new, e_new
    rho[Object_Mask] = rho0
    u[Object_Mask] = 0.0
    v[Object_Mask] = 0.0
    e[Object_Mask] = e0

    t = n * dt
    # Update plots elke paar stappen
    if n % 100 == 0:
        im1.set_clim([rho.min(), rho.max()])
        im2.set_clim([e.min(), e.max()])
        im3.set_clim([u.min(), u.max()])
        
        im1.set_data(rho.T)
        ax1.set_title(f"Density at t = {t:.5f}s")
        im2.set_data(e.T)
        ax2.set_title(f"Internal Energy at t = {t:.5f}s")
        im3.set_data(u.T)
        ax3.set_title(f"Speed at t = {t:.5f}s")
        plt.pause(0.0001)        

plt.ioff()

def main():
    plt.show()

if __name__ == "__main__":
    main()
