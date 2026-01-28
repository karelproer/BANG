import numpy as np
import matplotlib.pyplot as plt
from aerofoil import *
from derivatives import *
from simulation import Simluation
from matplotlib.widgets import Button, Slider

# Parameters
Nx, Ny = 300,300 #pixels
Lx, Ly = 1.0, 1.0 # groote van het gesimuleerde gebied
dx, dy = Lx / Nx, Ly / Ny
dt = 0.00001 # in seconden
Nt = 400000
gamma = 1.4
rho0, e0 = 1, 214_000
rot_deg = 0

# Rooster
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y, indexing='ij')

Object_Mask= object_mask(Nx, Ny, Lx, Ly, rot_deg, 1) # pixels waar de vleugel is

# simulatie object
sim = Simluation((Lx, Ly), (Nx, Ny), 1.4, 1, 214_000, (450, 0), Object_Mask)

# Visualisatie
plt.ion()
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16, 9)) # drie plaatjes

def on_close(event):
    exit()

fig.canvas.mpl_connect('close_event', on_close)


im1 = ax1.imshow(sim.rho.T, origin='lower', extent=[0, Lx, 0, Ly], cmap='inferno', vmin=0.9999, vmax=1.0001)
ax1.set_title("Density")
plt.colorbar(im1, ax=ax1)
im2 = ax2.imshow(sim.e.T, origin='lower', extent=[0, Lx, 0, Ly], cmap='inferno')
ax2.set_title("Internal Energy")
plt.colorbar(im2, ax=ax2)
im3 = ax3.imshow(sim.u.T, origin='lower', extent=[0, Lx, 0, Ly], cmap='inferno', vmin=0, vmax=1)
ax3.set_title("Speed")
plt.colorbar(im3, ax=ax3)
im4 = ax4.imshow(sim.mach, origin='lower', extent=[0, Lx, 0, Ly], cmap='inferno', vmin=0, vmax=1.5)
ax4.set_title("Mach Number")
plt.colorbar(im4, ax=ax4)

text = fig.text(0.47, 0.1, "", ha='center', va='bottom', fontsize=12, color='blue') # text die lift en weerstand laat zien
axrandom = fig.add_axes([0.1, 0.06, 0.1, 0.04])
axreset = fig.add_axes([0.8, 0.06, 0.1, 0.04])
axspeed = fig.add_axes([0.3, 0.06, 0.4, 0.04])
axrot = fig.add_axes([0.3, 0.94, 0.4, 0.04])

def preset(i):
    def onpressed(_):
        global Object_Mask
        Object_Mask = object_mask(Nx, Ny, Lx, Ly, rot_deg, i)
        sim.setObjectMask(Object_Mask)
        sim.reset(None)
    return onpressed

def set_speed(val):
    sim.v0 = (val, 0)
    sim.reset(None)

def set_rot(val):
    global rot_deg
    global Object_Mask
    rot_deg = val
    Object_Mask = object_mask(Nx, Ny, Lx, Ly, rot_deg, -1)
    sim.setObjectMask(Object_Mask)
    sim.reset(None)

bnreset = Button(axreset, 'reset')
bnreset.on_clicked(sim.reset)
bnrandom = Button(axrandom, 'random')
bnrandom.on_clicked(preset(6))
sldspeed = Slider(axspeed, 'speed', 0, 500, valinit=343)
sldspeed.on_changed(set_speed)
sldrot = Slider(axrot, 'rotation', -180, 180, valinit=0)
sldrot.on_changed(set_rot)

bnwings = []
for i in range(5):
    axwings = fig.add_axes([i*0.15 + 0.15, 0.005, 0.1, 0.04])
    bnwings.append(Button(axwings, f'preset {i+1}'))
    bnwings[i].on_clicked(preset(i+1))


for n in range(Nt):
    sim.step(dt)

    t = sim.time
    # Update elke paar stappen
    if n % 10 == 0:
        # automatisch aanpassen kleuren aan uiterste waarden
        im1.set_clim([sim.rho.min(), sim.rho.max()])
        im2.set_clim([sim.e.min(), sim.e.max()])
        im3.set_clim([sim.u.min(), sim.u.max()])
        
        # plaatjes updaten met nieuwe waarden
        im1.set_data(sim.rho.T)
        ax1.set_title(f"Density at t = {t:.5f}s")
        im2.set_data(sim.e.T)
        ax2.set_title(f"Internal Energy at t = {t:.5f}s")
        im3.set_data(sim.u.T)
        ax3.set_title(f"Speed at t = {t:.5f}s")
        im4.set_data(sim.mach.T)
        ax4.set_title(f"Mach Number at t = {t:.5f}s")
        text.set_text(f"Drag = {sim.drag:.2f} N                               Lift = {sim.lift:.2f} N")
        
        #wachten voor laten zien
        plt.pause(0.0001)        
        plt.show()
        