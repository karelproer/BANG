import numpy as np

def object_mask(Nx, Ny, Lx, Ly, rot_deg):
    # Rooster
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    Object_Mask = np.zeros((Nx, Ny), dtype=bool)

    M = 0.06 # Max camber %
    P = 0.4  # Max camber position in %
    T = 0.12 # Max thickness
    a0, a1, a2, a3, a4 = 0.2969, -0.126, -0.3156, 0.2843, -0.1036 #standaard waarden

    # rotation angle
    alpha = np.deg2rad(rot_deg)

    # tip of wing before scaling
    x0, y0 = 0.3, 0.5

    # shift to center
    Xs = X - x0
    Ys = Y - y0

    # rotate grid into wing coordinates
    Xr =  Xs*np.cos(alpha) + Ys*np.sin(alpha)
    Yr = -Xs*np.sin(alpha) + Ys*np.cos(alpha)

    # scale
    x_wing = Xr / 0.5
    y_wing = Yr / 0.5
  
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
    return Object_Mask