import numpy as np
import matplotlib.pyplot as plt
from derivatives import *
from simulation import Simluation

# Parameters
Nx, Ny = 200,200
Lx, Ly = 1.0, 1.0
ax, ay = 0, 0.0
dx, dy = Lx / Nx, Ly / Ny
dt = 0.000001 # in seconden
Nt = 400000
gamma = 1.4
rho0, e0 = 1, 214_000

# Rooster
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y, indexing='ij')

Object_Mask = np.zeros((Nx, Ny), dtype=bool)

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




sim = Simluation((1.0, 1.0), (Nx, Ny), 1.4, 1, 214_000, (500, 0), Object_Mask)



# Visualisatie setup
plt.ion()
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5))
im1 = ax1.imshow(sim.rho.T, origin='lower', extent=[0, Lx, 0, Ly], cmap='inferno', vmin=0.9999, vmax=1.0001)
ax1.set_title("Density")
plt.colorbar(im1, ax=ax1)

im2 = ax2.imshow(sim.e.T, origin='lower', extent=[0, Lx, 0, Ly], cmap='inferno')
ax2.set_title("Internal Energy")
plt.colorbar(im2, ax=ax2)

im3 = ax3.imshow(sim.u.T, origin='lower', extent=[0, Lx, 0, Ly], cmap='inferno', vmin=0, vmax=1)
ax3.set_title("Speed")
plt.colorbar(im3, ax=ax3)

for n in range(Nt):
    sim.step(dt)

    #Fx, Fy = rho * ax, rho * ay

    t = n * dt
    # Update plots elke paar stappen
    if n % 100 == 0:
        im1.set_clim([sim.rho.min(), sim.rho.max()])
        im2.set_clim([sim.e.min(), sim.e.max()])
        im3.set_clim([sim.u.min(), sim.u.max()])
        
        im1.set_data(sim.rho.T)
        ax1.set_title(f"Density at t = {t:.5f}s")
        im2.set_data(sim.e.T)
        ax2.set_title(f"Internal Energy at t = {t:.5f}s")
        im3.set_data(sim.u.T)
        ax3.set_title(f"Speed at t = {t:.5f}s")
        plt.pause(0.001)        
        
plt.ioff()

def main():
    plt.show()
if __name__ == "__main__":
    main()
