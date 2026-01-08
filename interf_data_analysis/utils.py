import numpy as np
import astropy.units as u
import astropy.constants as con
from astropy.cosmology import FlatLambdaCDM
import matplotlib.pyplot as plt
cosmo = FlatLambdaCDM(H0=70.0, Om0=0.30) 


LINES = {
        'CO(14-13)': 1611.79,'CO(13-12)': 1496.92,'CO(12-11)': 1382.00,'CO(11-10)': 1267.01,'CO(10-9)': 1151.99, 
        'CO(9-8)': 1036.91, 'CO(8-7)': 921.800,'CO(7-6)': 806.652, 'CO(6-5)': 691.473, 'CO(5-4)': 576.268, 
        'CO(4-3)': 461.041, 'CO(3-2)': 345.796,'CO(2-1)': 230.538, 'CO(1-0)': 115.271, '[CI](2-1)': 809.342, 
        '[CI](1-0)': 492.161, 'HCN(4-3)':354.505478, 'H2O(211-202)':752.033143, 
}
LINES = dict(sorted(LINES.items(), key = lambda item: item[1]))

def convert_v(oldv, ref, to='vel',fro='freq'):
    '''convert between velocity (in km/s), redshift, and frequency (in GHz), using the radio convention'''
    c = con.c.to(u.km/u.s).value
    if to not in ['z','vel','freq'] or fro not in ['z','vel','freq']:
        raise ValueError(f'Invalid input/output velocity types: {to}, {fro}')
    if fro == 'z':
        if to == 'vel':
            newv = c*oldv/(1.+oldv) #v = cz/(1+z)
        else: # if to == 'freq'
            newv = ref/(1.+oldv) #f = f0/(1+z)
    elif fro == 'vel':
        if to == 'z':
            newv = oldv/(c-oldv)  # z = v/(c-v)
        else: #if to =='freq'
            newv = ref *(1.-oldv/c) #f = f0 * (1-v/c)
    else: #if fro == 'freq'
        if to == 'vel':
            newv = (ref-oldv)/ref * c # v = (f0-f)/f * c
        else: #if to == 'z'
            newv = ref/oldv - 1.0 # z=f0/f
    return newv


def get_z(freq, obs):
    return np.round(freq/obs-1,3)

def find_possible_zs(obs, verbose=False, co_only=False):
    '''Get possible redshifts for a particular observed line frequency.
    
    Parameters
    ----------
    obs (array): observed frequency of line in GHz
    
    Returns
    -------
    zdict: dictionary (f_obs, line, z) with observed frequencies, line names, and redshifts
    '''
    zdict={'f_obs':obs, 'line':[], 'z':[]}
    for line in LINES:
        if co_only:
            if 'CO' not in line: continue
        zi = get_z(LINES[line],obs)
        if zi < 8. and zi > 0.5:
            if verbose: print(line,': '+f'z = {zi:.3f}')
            zdict['line'].append(line)
            zdict['z'].append(zi)
    return zdict


def redshift2lines(z):
    dt = {}
    for line in LINES.keys():
        dt[line] = LINES[line] / (1.+z)
    return dt

def gauss(x, *p0):
    mu, sigma, N = p0
    return 10**N*1./np.sqrt(2*np.pi*sigma**2) * np.exp(-(x-mu)**2/(2*sigma**2))

def plot_map(mapi):
    '''convenience function to plot a map extracted from a cube (using correct transposition/axes)'''
    _,ax = plt.subplots()
    sc = ax.imshow(mapi.T, origin='lower')
    ax.colorbar(sc)
    plt.show()

def distance_array(map, center):
    '''create array of euclidean distance between array elements and "center" pixel (in numpy format, i.e., y, x)
    
    center=(px,py)
    map[py,px] = center_val
    '''
    x = np.arange(map.shape[0]) 
    y = np.arange(map.shape[1])
    Y,X = np.meshgrid(y,x)
    dist = np.sqrt((X-center[1])**2 + (Y-center[0])**2)
    return dist
