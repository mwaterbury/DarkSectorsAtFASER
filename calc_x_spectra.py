### HEADER
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from os import listdir
import re

import scipy.integrate as integrate

### Physical constants
c = 3 * 10 ** 8
hbar = 6.6 * 10 ** -16 # eV * s
hbar = hbar * 10 ** -9 # GeV * s

# Minkowski Metric
eta = np.diag([1,-1,-1,-1])

### Masses and Lifetimes

m_pi0 = 0.135 # GeV
m_eta = 0.548 # GeV
m_KL  = 0.497 # GeV
m_D   = 1.865 # GeV
m_b   = 4.180 # GeV
m_B   = 5.280 # GeV
m_mu  = 105.6583755 * 10 ** -3 # GeV
v     = 246 # GeV

tau_D = 4.101 * 10**-13 # s
tau_B = 1.519 * 10**-12 # s
tau_KL = 5.116 * 10**-8 # s
tau_eta = 5.04 * 10**-19 # s (Inverting width in PDG)
tau_pi0 = 8.43 * 10**-17 # s

### FASER Geometry

L_FASER = 480
Delta_FASER = 1.5
R_FASER = 0.1
theta_FASER = R_FASER / L_FASER

L_FASER2 = 480
Delta_FASER2 = 5
R_FASER2 = 1
theta_FASER2 = R_FASER2 / L_FASER2

### Functions to determine position of X from hadron decay

# Boosts along z-direction
def z_boost(gam):
    betagam = np.sqrt(gam**2 - 1)
    mat = [[gam, 0, 0, betagam],
           [0, 1, 0, 0],
           [0, 0, 1, 0],
           [betagam, 0, 0, gam]]
    return np.array(mat)

# Rotates in y-z plane
def z_rot(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    mat = [[1, 0, 0, 0],
           [0, 1, 0, 0],
           [0, 0, c, -s],
           [0, 0, s, c]]
    return np.array(mat)

# Determines rest momentum of X from H -> X + Y
def rest_momentum(m_H, m_X, m_Y=0, theta=0, phi=0):
    
    # From PDG on 2-body decay
    sigma = m_X ** 2 + m_Y ** 2
    delta = m_X ** 2 - m_Y ** 2
    pt = (m_H ** 4 - 2 * m_H ** 2 * sigma + delta ** 2) ** (1/2) / (2 * m_H)
    E  = (pt ** 2 + m_X ** 2) ** (1/2)
    px = pt * np.sin(theta) * np.sin(phi)
    py = pt * np.sin(theta) * np.cos(phi)
    pz = pt * np.cos(theta)
    
    return np.array([E,px,py,pz])

# Extracts anglular direction from a momentum
def extract_spherical(p):
    p = p[1:]
    n = p / np.dot(p.T,p)
    if n[1] == 0:
        phi = np.sign(n[0])*np.pi/2
    else:
        phi = np.arctan2(n[0],n[1])
    if n[2] == 0:
        theta = np.pi/2
    else:
        theta = np.arctan(np.sqrt(n[0]**2+n[1]**2)/n[2])
    theta = theta * np.sign(n[2])
    return theta, phi

# Determines the location of X in the plane at the front of FASER
def geometry_X(theta_H, p_H, m_X, L=L_FASER2, R=R_FASER2, Delta=Delta_FASER2,
                    tau_H = tau_B, m_H=m_B, m_Y=0, theta_r=0, phi_r=0):
    d = c * tau_H * p_H / m_H
    l = L - d * np.cos(theta_H) # Distance to travel to FASER at decay
    y = d * np.sin(theta_H) # y-displacement of decay point
    
    p_r = rest_momentum(m_H, m_X, m_Y, theta=theta_r, phi=phi_r)
    p_boost = np.dot(z_rot(-theta_H),np.dot(z_boost(p_H/m_H),p_r))
    theta2, phi2 = extract_spherical(p_boost)
    
    x0 = l * np.tan(theta2) * np.cos(phi2)
    y0 = y + l * np.tan(theta2) * np.sin(phi2)
    r0 = np.sqrt( x0 ** 2 + y0 ** 2 )

    x1 = (l+Delta) * np.tan(theta2) * np.cos(phi2)
    y1 = y + (l+Delta) * np.tan(theta2) * np.sin(phi2)
    r1 = np.sqrt( x1 ** 2 + y1 ** 2 )
    z1 = Delta
    
    t = 1
    delta = 0
    if np.sqrt(x1 ** 2 + y1 ** 2) > R:
        st = np.sin(theta2)
        ct = np.cos(theta2)
        cp = np.cos(phi2)
        sp = np.sin(phi2)
        
        a = (x0 * cp + y0 * sp) / (2*st)
        t = np.sqrt(a ** 2 + (R ** 2 - r0 ** 2)) - a

        x1 = x0 + st * cp * t
        y1 = y0 + st * sp * t
        z1 = ct * t
        r1 = np.sqrt(x1 ** 2 + y1 ** 2)
        
    if t > 0: delta = np.sqrt((x1-x0) ** 2 + (y1-y0) ** 2  + z1 ** 2)
    l = np.sqrt(l ** 2 + x0 ** 2 + (y-y0) ** 2)

    return {
            'p_H'     : p_H,
            'theta_H' : theta_H,
            'y_H'     : y,
            'p_X'     : p_boost,
            'theta_X' : theta2,
            'phi_X'   : phi2,
            'x0'      : x0,
            'y0'      : y0,
            'r0'      : r0,
            'x1'      : x1,
            'y1'      : y1,
            'r1'      : r1,
            'z1'      : z1,
            'delta'   : delta,
            'l'       : l,
        }

# Uses above functions to calculate the X spectrum from the B spectrum.
def main(m_Xs, m_Y):
    b_spectrum = np.genfromtxt('Spectra/bbbar_spectrum.csv',
                                delimiter=',',skip_header=1)
    
    thetas = np.sort(list(set(b_spectrum[:,1])))
    ps     = np.sort(list(set(b_spectrum[:,2])))
    
    for m_X in m_Xs:
        print('Generating data for m_X=' + str(m_X))
        x_geodata = []
        for p_H in ps:
            for theta_H in thetas:
                for theta_r in np.pi * np.logspace(-6, -0.1, 50):
                    for phi_r in 2*np.pi* np.linspace(0, 1, 50):
                        x_geo = geometry_X(theta_H = theta_H, p_H = p_H,
                                            theta_r = theta_r, phi_r = phi_r,
                                            m_X = m_X, m_Y = m_Y, m_H = m_B)
                        x_geo['theta_r'] = theta_r
                        x_geo['phi_r'] = phi_r
                        x_geodata.append(x_geo)

        np.save('Spectra/x_geo_b_mX_' + str(m_X) + '_mY_' + str(m_Y) + '.npy', np.array(x_geodata))

##########################################
# This code is called when script is ran #
#     via `python calc_x_spectra.py`     #
##########################################


m_Xs = [0.3, 0.6, 1.2, 2.4]
m_Y  = m_KL

x_geodata = main(m_Xs, m_Y)
