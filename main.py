import numpy as np
import matplotlib.pyplot as plt
from aerofoil import *
from derivatives import *
from simulation import Simluation

# Parameters
Nx, Ny = 300,300
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

Object_Mask= object_mask(Nx, Ny, Lx, Ly)

sim = Simluation((1.0, 1.0), (Nx, Ny), 1.4, 1, 214_000, (300, 0), Object_Mask)

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
