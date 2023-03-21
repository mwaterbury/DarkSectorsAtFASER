### HEADER
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from os import listdir
import re

import scipy.integrate as integrate

### Physical constants
c = 3 * 10 ** 8
hbar = 6.6 * 10 ** -16 # eV * s
hbar = hbar * 10 ** -9 # GeV * s

# Minkowski Metric
eta = np.diag([1,-1,-1,-1])

# PDG Values
m_B = 5.280
m_KL = 0.497
tau_B = 1.519 * 10**-12 # s

### Functions to determine position of X from hadron decay

# boosts along z-direction
def z_boost(gam):
    betagam = np.sqrt(gam**2 - 1)
    mat = [[gam, 0, 0, betagam],
           [0, 1, 0, 0],
           [0, 0, 1, 0],
           [betagam, 0, 0, gam]]
    return np.array(mat)

# rotates in y-z plane
def z_rot(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    mat = [[1, 0, 0, 0],
           [0, 1, 0, 0],
           [0, 0, c, -s],
           [0, 0, s, c]]
    return np.array(mat)

# determines rest momentum of X from H -> X + Y
def rest_momentum(m_1, m_2, m_3):
    
    # From PDG on 2-body decay
    sigma = m_2 ** 2 + m_3 ** 2
    delta = m_2 ** 2 - m_3 ** 2
    pt = (m_1 ** 4 - 2 * m_1 ** 2 * sigma + delta ** 2) ** (1/2) / (2 * m_1)
    E2  = (pt ** 2 + m_2 ** 2) ** (1/2)
    E3  = (pt ** 2 + m_3 ** 2) ** (1/2)
    
    return (E2,E3,pt)

# extracts anglular direction from a momentum
def extract_spherical(p):
    p = p[1:]
    mag = np.sqrt(np.dot(p.T,p))
    n = p / mag
    phi = np.arctan2(n[1],n[0])
    if phi < 0:
        phi = phi + 2*np.pi
    theta = np.arctan2(np.sqrt(n[0]**2+n[1]**2),n[2]) 
    theta = np.abs(theta)
    return np.array([mag, theta, phi])

# determines the angular distribution of decay products
def boosted_4momentum_distribution(p_1,theta_1,m_1,m_2,m_3, theta_r=None, phi_r=None):
    
    if (theta_r is None or phi_r is None):
        eps = 10**-4
        theta_r = np.linspace(-1+eps,1-eps,50)
        theta_r = np.arccos(theta_r)
        phi_r = 2 * np.pi * np.linspace(eps,1-eps,100)

    E2, E3, pt = rest_momentum(m_1, m_2, m_3)

    st = np.sin(theta_r)
    ct = np.cos(theta_r)
    sp = np.sin(phi_r)
    cp = np.cos(phi_r)

    units = np.array([np.outer(st, cp ),
                      np.outer(st, sp ),
                      np.outer(ct, np.ones(np.shape(phi_r)))])
    t_units = np.ones((1,np.size(theta_r),np.size(phi_r)))
    p2s = np.concatenate((E2*t_units,  pt*units), axis=0)
    p3s = np.concatenate((E3*t_units, -pt*units), axis=0)
    
    angles = [[i,j] for i in theta_r for j in phi_r]
    angles = np.reshape(np.array(angles), (np.size(theta_r),np.size(phi_r),2))
    angles = np.array([angles[:,:,0], angles[:,:,1]])

    gam = np.sqrt(p_1 ** 2 + m_1 ** 2) / m_1
    xform = np.dot(z_rot(theta_1),z_boost(gam))

    p2s_xformed = np.einsum('ij,jmn->imn',xform,p2s)
    p3s_xformed = np.einsum('ij,jmn->imn',xform,p3s)

    return (angles, p2s_xformed, p3s_xformed)

def boosted_angular_distribution(p_1, theta_1, m_1, m_2, m_3, theta_r=None, phi_r=None):

    angles, p2s, p3s = boosted_4momentum_distribution(p_1, theta_1, m_1, m_2, m_3, theta_r, phi_r)

    p2s_angles = np.apply_along_axis(extract_spherical,axis=0,arr=p2s)
    p3s_angles = np.apply_along_axis(extract_spherical,axis=0,arr=p3s)

    return angles, p2s_angles, p3s_angles

###########################################################################
###########################################################################
###########################################################################
############################ Convolute Spectrum ###########################
###########################################################################
###########################################################################
###########################################################################

def spectrum_from_data_row(data, m_1, m_2, m_3, theta_r, phi_r):

    angles, p2s_angles, p3s_angles = boosted_angular_distribution(data[1], data[0], m_1, m_2, m_3,
                                                            theta_r, phi_r)

    theta_rs = np.ndarray.flatten(angles[0,:,:])
    phi_rs   = np.ndarray.flatten(angles[1,:,:])
    
    p2s     = np.ndarray.flatten(p2s_angles[0,:,:])
    theta2s = np.ndarray.flatten(p2s_angles[1,:,:])
    phi2s   = np.ndarray.flatten(p2s_angles[2,:,:])

    p3s     = np.ndarray.flatten(p3s_angles[0,:,:])
    theta3s = np.ndarray.flatten(p3s_angles[1,:,:])
    phi3s   = np.ndarray.flatten(p3s_angles[2,:,:])

    p1s     = np.ones(np.shape(p2s)) * data[1]
    theta1s = np.ones(np.shape(p2s)) * data[0]
    xsec    = np.ones(np.shape(p2s)) * data[2] / np.size(p2s)

    return np.stack((p1s, theta1s, xsec, theta_rs, phi_rs,
                     p2s, theta2s, phi2s, p3s, theta3s, phi3s,
                     )).T

def conv_spectrum(spectrum, m_1, m_2, m_3, theta_r = None, phi_r = None):

    function = lambda x : spectrum_from_data_row(x, m_1, m_2, m_3, theta_r, phi_r)

    convoluted = np.apply_along_axis(function,axis=1,arr=spectrum)

    # flatten from 3d array to 3d array
    convoluted = np.stack((np.ndarray.flatten(convoluted[:,:,0]),
                           np.ndarray.flatten(convoluted[:,:,1]),
                           np.ndarray.flatten(convoluted[:,:,2]),
                           np.ndarray.flatten(convoluted[:,:,3]),
                           np.ndarray.flatten(convoluted[:,:,4]),
                           np.ndarray.flatten(convoluted[:,:,5]),
                           np.ndarray.flatten(convoluted[:,:,6]),
                           np.ndarray.flatten(convoluted[:,:,7]),
                           np.ndarray.flatten(convoluted[:,:,8]),
                           np.ndarray.flatten(convoluted[:,:,9]),
                           np.ndarray.flatten(convoluted[:,:,10]),
                        )).T

    return convoluted

###########################################################################
###########################################################################
###########################################################################
########################### Decay Probabilities ###########################
###########################################################################
###########################################################################
###########################################################################

def decay_geometry(data, L, R, Delta, m = m_B, tau = tau_B):
    p_B = data[0]
    t_B = data[1]
    p_X = data[5]
    t_X = data[6]
    a_X = data[7]
    
    c_B = np.cos(t_B)
    s_B = np.sin(t_B)
    
    c_X = np.cos(t_X)
    s_X = np.sin(t_X)
    ca_X = np.cos(a_X)
    sa_X = np.sin(a_X)
    
    decay_length = ( np.sqrt(p_B ** 2 + m ** 2) / m ) * c * tau # gamma * c * tau 
    decay_point = decay_length * np.array([0, np.sin(t_B), np.cos(t_B)])
    L_eff = L - decay_point[2]
    
    X_trajectory = np.array([s_X * ca_X, s_X * sa_X, c_X])
    entry_point = decay_point + L_eff / c_X * X_trajectory 
    r2_entry = entry_point[0] ** 2 + entry_point[1] ** 2
    
    if r2_entry >= R ** 2:
        return 0, 0
    
    Delta_eff = Delta / c_X
    exit_point = entry_point + Delta_eff * X_trajectory
    r2_exit = exit_point[0] ** 2 + exit_point[1] ** 2
    
    if r2_exit < R ** 2:
        return L_eff, Delta_eff
    
    Delta_eff_transverse2 = R**2 - r2_entry
    Delta_eff_long2       = Delta_eff_transverse2 / s_X ** 2
    Delta_eff = np.sqrt(Delta_eff_long2 + Delta_eff_transverse2)
    return L_eff, Delta_eff

def prob_decay_in_volume(p, m, ctau, L, Delta):
    d = ( np.sqrt(p ** 2 + m ** 2) / m ) * ctau # gamma * ctau
    return np.exp(-L / ctau) * ( 1 - np.exp( - Delta / ctau ))

def decay_prob_from_row(data, ctau, L, R, Delta, m_H, tau_H, m_X):
    
    # Find geometry of event to feed into decay probability
    L_eff, Delta_eff = decay_geometry(data, L, R, Delta, m_H, tau_H)
    
    # Determine decay probability
    return prob_decay_in_volume(data[5], m_X, ctau, L_eff, Delta_eff)

def calculate_decay_prob(dataset, ctau, L, R, Delta, m_X, m_H = m_B, tau_H = tau_B):
    function = lambda x : decay_prob_from_row(x, ctau, L, R, Delta, m_H, tau_H, m_X)
    # Apply to each row of dataset
    return np.apply_along_axis(function, axis=1, arr=dataset)
    
def calculate_decay_xsec(dataset, ctau, L, R, Delta, m_X, m_H = m_B, tau_H = tau_B):
    probabilities = calculate_decay_prob(dataset, ctau, L, R, Delta, m_H, tau_H, m_X)
    return np.sum(dataset[:,2] * probabilities) # xsec * prob

###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################

def test(p1, theta1, m1, m2, m3, theta_r = None, phi_r = None):
    angles, p2s, p3s = boosted_angular_distribution(p1, theta1, m1, m2, m3,
                                                  theta_r, phi_r)

    ps = np.ndarray.flatten(p2s[0,:,:])
    thetas = np.ndarray.flatten(p2s[1,:,:])
    phis = np.ndarray.flatten(p2s[2,:,:])
    print(np.min(ps), np.max(ps))
    print(np.min(thetas),np.max(thetas))
    print(np.min(phis),np.max(phis))

    fig = plt.figure()
    ax = fig.add_subplot(1,3,1)
    ax.hist(ps, bins=10)
    ax = fig.add_subplot(1,3,2)
    ax.hist(thetas, bins=np.pi*np.logspace(-6,0,200))
    # ax.set_xlim([0,np.pi])
    ax.set_xscale('log')
    ax = fig.add_subplot(1,3,3)
    ax.hist(phis, bins=phi_r)
    ax.set_xlim([0,2*np.pi])
#     ax.set_xscale('log')

    plt.figure()
    h = plt.hist2d(thetas, phis, bins=[np.pi*np.logspace(-6,0,200), 50])
    # plt.xlim([0,np.pi])
    plt.xscale('log')
    plt.ylim([0,2*np.pi])
    plt.colorbar()
    plt.show()
    return h

def plot_dataset(dataset, i, j, log=False):
    plt.figure()
    p_Bs     = dataset[:,0]
    theta_Bs = dataset[:,1]
    theta_rs = dataset[:,3]
    phi_rs   = dataset[:,4]
    xsecs    = dataset[:,2]
    xsecs_logscaled = xsecs * p_Bs * theta_Bs * np.log(10) ** 2
    if log:
        xsecs_logscaled = xsecs_logscaled * theta_rs * phi_rs * np.log(10) ** 2
    x     = dataset[:,i]
    y     = dataset[:,j]

    x_min = np.min(x)
    x_max = np.max(x)
    y_min = np.min(y)
    y_max = np.max(y)

    x_bins = np.logspace(np.log10(x_min), np.log10(x_max))
    y_bins = np.logspace(np.log10(y_min), np.log10(y_max))

    plt.hist2d(x, y, bins=[x_bins, y_bins],
                   weights = xsecs_logscaled,
                   norm=matplotlib.colors.LogNorm())
    plt.colorbar()
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################

def main():
    eps = 10**-4
    theta_r =     (1-eps) * np.logspace(-6, 0, 100)
    theta_r = np.append(theta_r-1, 1-theta_r)
    # phi_r   = 2 * (np.pi-eps) * np.logspace(-6, 0, 50)

    theta_r = np.arccos(theta_r)
    phi_r = 2 * np.pi * np.linspace(eps,1-eps,50)
    # plt.xscale('log')
    
    # theta_r = None
    # phi_r   = None

    if (theta_r is None or phi_r is None):
        eps = 10**-4
        theta_r = np.linspace(-1+eps,1-eps,100)
        theta_r = np.arccos(theta_r)
        phi_r = 2 * np.pi * np.linspace(eps,1-eps,200)
    
    b_spectrum = np.genfromtxt('Spectra/bbbar_spectrum.csv',
                                delimiter=',', skip_header=1)
    # Extract theta, p, xsec from spectrum
    b_spectrum = np.stack((b_spectrum[:,1], b_spectrum[:,2],
                           b_spectrum[:,10])).T

    # for m_X in [0.3, 0.6, 1.2, 2.0, 2.4]:
    for m_X in [2.0]:
        print('Calculating for m_X = ' + str(m_X))
        decay_spectrum = conv_spectrum(b_spectrum, m_B, m_X, m_KL,
                                       theta_r, phi_r)
        
        np.save('Spectra/X_' + str(m_X) + '_Y_' + str(m_KL) +
                    '_from_bspectrum_logsample.npy',decay_spectrum)
        print('Finished m_X = ' + str(m_X))

# main()
        
# m_X = 2.4
# eps = 10**-4
# theta_r = (np.pi - eps) * np.logspace(-6, 0, 50)
# phi_r = 2 * (np.pi - eps) * np.logspace(-6, 0, 50)

# b_spectrum = np.genfromtxt('Spectra/bbbar_spectrum.csv',
#                        delimiter=',', skip_header=1)
# # Extract theta, p, xsec from spectrum
# b_spectrum = np.stack((b_spectrum[:,1], b_spectrum[:,2], b_spectrum[:,10])).T

# linear_dataset = conv_spectrum(b_spectrum, m_B, m_X, m_KL)
# log_dataset    = conv_spectrum(b_spectrum, m_B, m_X, m_KL, theta_r, phi_r)

# plot_dataset(linear_dataset, 6, 5)
# plot_dataset(log_dataset, 6, 5, log=True)
# plt.show()

eps = 10**-4
theta_r =     (1-eps) * np.logspace(-6, 0, 50)
theta_r = np.append(theta_r-1, 1-theta_r)
# phi_r   = 2 * (np.pi-eps) * np.logspace(-6, 0, 50)

theta_r = np.arccos(theta_r)
phi_r = 2 * np.pi * np.linspace(eps,1-eps,100)

# # theta_r = None
# # phi_r = None

test(100, 10**-2, 5, 1, 0.5, theta_r, phi_r)