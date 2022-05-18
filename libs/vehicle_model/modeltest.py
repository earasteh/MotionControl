import numpy as np
from numpy import cos, sin, sqrt, arctan, pi
import matplotlib.pyplot as plt
from libs.vehicle_model.vehicle_model import VehicleParameters

"""
Tire model test
"""
p1 = VehicleParameters()


def full_tire_function(p, lat_slip, long_slip, delta):
    ## Longitudinal slip
    sFLx, sFRx, sRLx, sRRx = long_slip

    ## Lateral slip
    sFLy, sFRy, sRLy, sRRy = lat_slip

    ## delta
    deltaFL, deltaFR, deltaRL, deltaRR = delta

    # Parameters
    g = 9.81

    p.DFL = 1
    p.DFR = 1
    p.DRL = 1
    p.DRR = 1

    ## Normal forces (static forces)
    fFLz0 = p.b / (p.a + p.b) * p.m * g / 2
    fFRz0 = p.b / (p.a + p.b) * p.m * g / 2
    fRLz0 = p.a / (p.a + p.b) * p.m * g / 2
    fRRz0 = p.a / (p.a + p.b) * p.m * g / 2

    fFLz = fFLz0
    fFRz = fFRz0
    fRLz = fRLz0
    fRRz = fRRz0

    ## Combined slip
    sFL = sqrt(sFLx ** 2 + sFLy ** 2)
    sFR = sqrt(sFRx ** 2 + sFRy ** 2)
    sRL = sqrt(sRLx ** 2 + sRLy ** 2)
    sRR = sqrt(sRRx ** 2 + sRRy ** 2)

    # Compute tire forces
    ## Combined friction coefficient
    muFL = p.DFL * sin(p.CFL * arctan(p.BFL * sFL))
    muFR = p.DFR * sin(p.CFR * arctan(p.BFR * sFR))
    muRL = p.DRL * sin(p.CRL * arctan(p.BRL * sRL))
    muRR = p.DRR * sin(p.CRR * arctan(p.BRR * sRR))

    ## Lateral Friction coefficient
    if sFL != 0:
        muFLy = sFLy * muFL / sFL
    else:
        muFLy = p.DFL * sin(p.CFL * arctan(p.BFL * sFLy))

    if sFR != 0:
        muFRy = sFRy * muFR / sFR
    else:
        muFRy = p.DFR * sin(p.CFR * arctan(p.BFR * sFRy))

    if sRL != 0:
        muRLy = sRLy * muRL / sRL
    else:
        muRLy = p.DRL * sin(p.CRL * arctan(p.BRL * sRLy))

    if sRR != 0:
        muRRy = sRRy * muRR / sRR
    else:
        muRRy = p.DRR * sin(p.CRR * arctan(p.BRR * sRRy))

    ## Compute lateral forces
    fFLyt = muFLy * fFLz
    fFRyt = muFRy * fFRz
    fRLyt = muRLy * fRLz
    fRRyt = muRRy * fRRz

    ## Rotate to obtain forces in the chassis frame
    fFLy = + fFLyt * cos(deltaFL) + fFRyt * cos(deltaFR) # total front
    fRLy = + fRLyt * cos(deltaRL)
    fRRy = + fRRyt * cos(deltaRR)

    return fFLy


def simplified_tire_function(p, alpha_f):
    fFLz0 = p.b / (p.a + p.b) * p.m * 9.81 / 2
    Ffy = 2 * p.DFL * sin(p.CFL * np.arctan(p.BFL * alpha_f)) * fFLz0
    return Ffy


long_slip = [0.0] * 4
delta = np.linspace(-7 * pi / 180, 7 * pi / 180, 1000)

Fy_full = []
Fy_simple = []

for d in delta:
    Fy_full.append(full_tire_function(p1, [d] * 4, long_slip, [d] * 4))
    Fy_simple.append(simplified_tire_function(p1, d))

plt.figure()
plt.plot(delta * 180 / np.pi, Fy_full, 'r')
plt.plot(delta * 180 / np.pi, Fy_simple, 'b')
plt.xlabel(r'$\alpha (degrees)$')
plt.ylabel(r'$F_y (N)$')
plt.legend(['Normal values', 'MPC values'])
plt.grid()
plt.show()
