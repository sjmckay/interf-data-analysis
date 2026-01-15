import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from .array_utils import distance_array
from interferopy import tools as iftools
from astropy.stats import sigma_clip


def gauss(x, *p0):
    mu, sigma, N = p0
    return 10**N*1./np.sqrt(2*np.pi*sigma**2) * np.exp(-(x-mu)**2/(2*sigma**2))

def plot_hist(linemap, mask=None):
    f,ax = plt.subplots()
    nanmask = np.isnan(linemap)|(np.abs(linemap)>0.2)
    if mask is not None:
        nanmask != mask
    bars,edges,_ = ax.hist(linemap[~nanmask],bins=101)
    edges=edges[0:-1]
    popt, perr = curve_fit(xdata=edges[edges<3], ydata=bars[edges<3], f=gauss, p0=(0.0,2.0,7.0))
    
    ax.set_yscale('log')
    ax.plot(edges, gauss(edges,*popt))
    ax.annotate(f'Sigma = {popt[1]:.3f}',(0.04,0.95),xycoords='axes fraction')
    plt.show()
    return popt[1] # return sigma of Gaussian fit


def get_channel_err(im_uncor, pbspec=None, beam_corr=1.0): 
    """Determine rms error in cube by channel, similar to interferopy get_err function but with modifications
    
    Parameters
    ----------
    im_uncor (array): uncorrected image cube (y,x,chan)
    pbspec (array): primary beam spectrum (chan), or None. If None, no pb spec correction is done.
    beam_corr (float): correction factor for beam size to multiply raw error by (e.g., to account for aperture size)
    
    Returns
    -------
    err (array): error in each channel of cube
    """
    err = np.zeros(im_uncor.shape[2])
    for i in range(im_uncor.shape[2]):
        err[i] = iftools.calcrms(im_uncor[:,:,i])
    if pbspec is None: err = err/pbspec
    err *= beam_corr
    return err



def compute_2d_rms(map, exclude_r=None, mask=None, sigma=3.0, maxiters=5,copy=True):
    """get rms of 2d linemap, using optional mask and exclusion radius (inside which will be blanked out)"""
    if copy: nmap = map.copy()
    else: nmap=map
    center = nmap.shape[1] / 2.0, nmap.shape[0] / 2.0
    
    # make exclusion mask
    dist = distance_array(nmap, center)
    exclusion_mask = np.zeros_like(nmap, dtype=bool)
    if exclude_r is not None:
        exclusion_mask |= dist < exclude_r
    if mask is not None:
        exclusion_mask |= mask
    
    # Apply exclusion mask
    data = nmap[~exclusion_mask]
    # Drop NaNs and zeros 
    data = data[~np.isnan(data)]
    data = data[data != 0]
    clipped = sigma_clip(data, sigma=sigma, maxiters=maxiters, stdfunc=np.nanstd)
    rms = np.nanstd(clipped.data[~clipped.mask])
    nsamples = len(clipped.data[~clipped.mask])
    return rms, nsamples