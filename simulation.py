import numpy as np
from derivatives import *

class Simluation:
    def __init__(self, size, pixels, gamma, rho0, e0, v0, object_mask):
        # variabelen initialiseren
        self.size = size
        self.pixels = pixels
        self.gamma = gamma
        self.rho0 = rho0
        self.e0 = e0
        self.object_mask = object_mask

        #normals wijzen van buiten het obkect naar het object
        self.normals = []
        
        for i in range(1, pixels[0]-1):
            for j in range(1, pixels[1] - 1):
                if not object_mask[i, j]:
                    # Als het niet in het object zit maar buur wel dan is er een normal
                    if object_mask[i-1, j]: self.normals.append((i, j, -1, 0))   # links
                    if object_mask[i+1, j]: self.normals.append((i, j, +1, 0))   # rechts
                    if object_mask[i, j-1]: self.normals.append((i, j, 0, -1))   # naar beneden
                    if object_mask[i, j+1]: self.normals.append((i, j, 0, +1))   # omhoog
        

        self.mu = 1.8e-5        # Viscositeit van lucht
        self.Pr = 0.72          # Prandtl getal van lucht (hoeveelheid warmtestroom per viscositeit)
        self.kappa = self.mu * self.gamma / self.Pr # Absolute hoeveelheid warmtestroom

        # hoelang sinds hbegin simualtie
        self.time = 0

        # simulatiestaat initialiseren
        self.rho = np.ones(pixels) * rho0
        self.u = np.ones(pixels) * v0[0]
        self.v = np.ones(pixels) * v0[1]
        self.uv = np.sqrt((np.square(self.u) + np.square(self.v)))
        self.e = np.ones(pixels) * e0
        self.p = np.multiply(self.rho, self.e) * (self.gamma - 1)
        self.uv = np.sqrt((np.square(self.u) + np.square(self.v)))   
         
        self.c = np.sqrt(np.clip(self.gamma*self.p/np.clip(self.rho, a_min=0.1, a_max=100), a_min=0, a_max=1000_000_000))
        self.mach = self.uv / self.c

    # tuple met dx en dy erin
    @property
    def dxy(self):
        return self.size[0] / self.pixels[0], self.size[1] / self.pixels[1]
    
    # berekent de afgeleiden van de simulatie-staat volgens de Euler-vergelijkingen (geen viscositeit)
    def deuler(self, vars):
        #vars is een lijst met simulatie-staat erin
        rho, u, v, e = vars[0], vars[1], vars[2], vars[3]
        p = np.multiply(rho, e) * (self.gamma - 1)

        #afgeleiden berekenen
        drho_dx, drho_dy = ddxy(rho, self.dxy)
        du_dx, du_dy = ddxy(u, self.dxy)
        dv_dx, dv_dy = ddxy(v, self.dxy)
        dp_dx, dp_dy = ddxy(p, self.dxy)
        de_dx, de_dy = ddxy(e, self.dxy)

        div_u = du_dx + dv_dy

        Fx, Fy = rho * 0, rho * 0

        # Euler-vergelijkingen in Afgelijden-vorm 
        drho_dt = -(self.u * drho_dx + v * drho_dy + rho * div_u)
        du_dt = -(self.u * du_dx + v * du_dy + dp_dx / rho - Fx)
        dv_dt = -(self.u * dv_dx + v * dv_dy + dp_dy / rho - Fy)
        de_dt = -(self.u * de_dx + v * de_dy + p / rho * div_u)

        return np.array([drho_dt, du_dt, dv_dt, de_dt])

    # berekent de afgeleiden van de simulatie-staat volgens de Navier-Stokes-vergelijkingen (wel viscositeit, maar grotere afrondingsfouten)
    def d(self, vars):
        rho, u, v, e = vars[0], vars[1], vars[2], vars[3]
        p = np.multiply(rho, e) * (self.gamma - 1)

        # Afgeleiden
        drho_dx, drho_dy = ddxy(rho, self.dxy)
        du_dx, du_dy = ddxy(u, self.dxy)
        dv_dx, dv_dy = ddxy(v, self.dxy)
        dp_dx, dp_dy = ddxy(p, self.dxy)
        de_dx, de_dy = ddxy(e, self.dxy)

        div_u = du_dx + dv_dy

        laplace_u = laplacian(u, self.dxy)
        laplace_v = laplacian(v, self.dxy)
        laplace_e = laplacian(e, self.dxy)
        
        ddiv_u_dx, ddiv_u_dy = ddxy(div_u, self.dxy)

        # Viscose krachten volgens Stokes' hypothese
        viscous_u_force = (self.mu / rho) * (laplace_u + (1/3) * ddiv_u_dx)
        viscous_v_force = (self.mu / rho) * (laplace_v + (1/3) * ddiv_u_dy)
        
        # Warmte geleiding
        viscous_e_term = (self.kappa / rho) * laplace_e
        
        Fx, Fy = rho * 0, rho * 0

        drho_dt = -(self.u * drho_dx + v * drho_dy + rho * div_u)   
        
        # Euler vergelijkingen + Viscose krachten
        du_dt = -(self.u * du_dx + v * du_dy + dp_dx / rho - Fx) + viscous_u_force
        dv_dt = -(self.u * dv_dx + v * dv_dy + dp_dy / rho - Fy) + viscous_v_force
        
        # Euler vergelijking + warmte-geleiding
        de_dt = -(self.u * de_dx + v * de_dy + p / rho * div_u) + viscous_e_term
    
        return np.array([drho_dt, du_dt, dv_dt, de_dt])

    # Neemt een gewogen gemiddelde van vier verschillende tijdstappen
    def rk4(self, u, h):
        # moet beter zijn dan direct euler
        k1 = self.d(u)
        k2 = self.d(u + h*k1/2)
        k3 = self.d(u + h*k2/2)
        k4 = self.d(u + h*k3)
        return u + h/6*(k1 + 2*k2 + 2+k3 + k4)

    # Ga een tijdstap verder in de simulaties
    def step(self, dt):
        # maar een lijst van de simulatie states
        variables = np.array([self.rho, self.u, self.v, self.e])

        # bereken druk
        self.p = np.multiply(self.rho, self.e) * (self.gamma - 1)

        # Gebruikt rk4 om de veranderingen te berekenen en pas de veranderingen toe
        variables = self.rk4(variables, dt)
        rho_new, u_new, v_new, e_new = variables[0], variables[1], variables[2], variables[3]

        # snelheid van de lucht en lokale geluidssnelheid
        self.uv = np.sqrt((np.square(u_new) + np.square(v_new)))    
        self.c = np.sqrt(np.clip(self.gamma*self.p/np.clip(rho_new, a_min=0.1, a_max=100), a_min=0, a_max=1000_000_000))

        # numerieke diffuse om de simulatie stabiel te houden
        nu = 0.05 * self.dxy[0] * (self.uv + self.c)
        for m in [rho_new, u_new, v_new, e_new]:
            laplacian = np.zeros_like(m)
            
            laplacian[1:-1, 1:-1] = (m[2:, 1:-1] + m[:-2, 1:-1] + m[1:-1, 2:] + m[1:-1, :-2] - 4 * m[1:-1, 1:-1])

            # waarden aan randen
            laplacian[0, 1:-1] = (m[1, 1:-1] + m[1, 1:-1] + m[0, 2:] + m[0, :-2] - 4 * m[0, 1:-1]) # f[i-1] is effectively f[1]
            laplacian[-1, 1:-1] = (m[-2, 1:-1] + m[-2, 1:-1] + m[-1, 2:] + m[-1, :-2] - 4 * m[-1, 1:-1]) # f[i+1] is effectively f[-2]
            laplacian[1:-1, 0] = (m[1:-1, 1] + m[1:-1, 1] + m[2:, 0] + m[:-2, 0] - 4 * m[1:-1, 0])
            laplacian[1:-1, -1] = (m[1:-1, -2] + m[1:-1, -2] + m[2:, -1] + m[:-2, -1] - 4 * m[1:-1, -1])
            
            laplacian[:, 0] = laplacian[:, 1] # Outflow for the diffusion itself
            laplacian[:, -1] = laplacian[:, -2]
            
            # Pas diffusie toe
            m += nu * laplacian

        # Stel waarde van randen in
        for m in [rho_new, u_new, v_new, e_new]:
            m[0, :] = m[1, :]
            m[-1, :] = m[-2, :]
            m[:, 0] = m[:, 1]
            m[:, -1] = m[:, -2]
        # zet de waarden naar standaard waar de vleugel zit (dit doet hetzelfde als een echte vleugel in het echt zou doen)        
        self.rho, self.u, self.v, self.e = rho_new, u_new, v_new, e_new
        self.rho[self.object_mask] = self.rho0
        self.u[self.object_mask] = 0.0
        self.v[self.object_mask] = 0.0
        self.e[self.object_mask] = self.e0
        
        # bereken de lift en weerstand
        # druk waar de normals zitten zorgt voor kracht in de richting van de normals (viscose weerstand wordt genegeerd)
        self.drag, self.lift = 0, 0
        for (i, j, dx, dy) in self.normals:
            self.drag += self.p[i][j] * dx * self.dxy[0]
            self.lift += self.p[i][j] * dy * self.dxy[1]

        self.mach = self.uv / self.c
