import numpy as np
import pandas as pd
pd.set_option('display.max_rows', None)
import interferopy.tools as iftools
import os
from scipy.ndimage import correlate1d
from joblib import Parallel, delayed, parallel_backend

from astropy.coordinates import SkyCoord as SC
from astropy.io import fits
fits.conf.verify = 'silentfix'
from astropy.wcs import wcs, FITSFixedWarning
import astropy.units as u
from astropy.nddata import Cutout2D
from astropy.cosmology import FlatLambdaCDM
from astropy.stats import sigma_clip
cosmo = FlatLambdaCDM(H0=70.0, Om0=0.30) 

import warnings
warnings.filterwarnings("ignore", category=FITSFixedWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from .array_utils import distance_array, rebin, find_map_peak, load_cube, copy_sliced_cube
from .noise import compute_2d_rms, get_channel_err
from .redshift_utils import convert_v, sig2fwhm_kms, fwhm_kms2sig

def rebin_spectrum_err(freqs, spectrum, err, rebin_factor):
    '''Rebin a spectrum using averaging. 
    
    Parameters
    ----------
    freqs (array): frequencies of channels in GHz

    spectrum (array): spectrum values at given frequencies/channels

    err (array): error in spectrum values

    rebin_factor (float): factor by which to rebin the spectrum (i.e. rebin_factor = 2 will average every two spectral points).
        If this number does not divide len(freqs) evenly, the remainder will be dropped.


    Returns
    -------
    freq_rebin (array): rebinned frequencies (i.e., every rebin_factor frequencies)
    
    spec_rebin (array): rebinned spectrum (averaged)
    
    err_rebin (array): rebinned errors (summed in quadrature then downscaled by sqrt(rebin_factor))
    '''
    spec_rebin = rebin(spectrum,factor=rebin_factor,function='mean')
    err_rebin = rebin(err,factor=rebin_factor,function='quad')
    # each freq should be at center of bin, so take steps of f, the rebinning factor, from the f/2-th bin as far as 
    # one can neatly fit the data in the rebinned shape
    freq_rebin = freqs[rebin_factor//2:(len(freqs)//rebin_factor)*rebin_factor:rebin_factor] 
    return  freq_rebin, spec_rebin, err_rebin
    
    
def flux_snr_2D(linemap, wcsi, coord=None, rms_r = 10, search_r = 1):
    '''get location and flux of peak in line map'''
    ppas = np.abs(1.0/wcsi.pixel_scale_matrix[0,0]/3600)
    center_cut = wcsi.world_to_pixel(coord)
    dist = distance_array(linemap,center_cut)
    mask = dist>search_r*ppas
    py,px = find_map_peak(linemap,mask=mask) 
    peakflux = linemap[py,px] # double check order of axes
    rmsmask = dist>rms_r*ppas
    rms,nsamples = compute_2d_rms(linemap,exclude_r=2.0*ppas, mask = rmsmask)
    snr = peakflux/rms
    ra,dec = wcsi.all_pix2world(px,py,0) #pass (ra,dec) --> (ra,dec)
    newcoord = SC(ra, dec, unit='deg')

    return newcoord, (px,py), peakflux, rms, snr, nsamples 


def aperture_spectrum(cub, w=None, coord=None, r=0.5, pbim=None, pbspec=None):
    '''wrapper function for measuring a spectrum and err in a circular aperture, using interferopy.cube.Cube'''
    if coord is not None: ra,dec=coord.ra, coord.dec
    else: ra, dec = None, None
    if not hasattr(cub,'im'): #if not dealing with an interferopy.Cube
        if w is None:
            raise Exception('Either a Cube or a WCS object must be supplied.')
        spectrum, err = get_spec_err(cub, w=w, ra=ra, dec=dec, pbim=pbim, pbspec=pbspec)
    else: 
        err  = get_channel_err(cub,r=r,ra=ra,dec=dec,pbim=pbim, pbspec=pbspec)
        spectrum = cub.aperture_value(ra=ra, dec=dec, radius=r, calc_error=False)
    return spectrum, err


def gaussian_kernel(sigma, radius, norm):
    """
    Computes a 1-D Gaussian convolution kernel. modifed version of scipy _gaussian_kernel1d
    """
    sigma2 = sigma * sigma
    x = np.arange(-radius, radius+1)
    phi_x = np.exp(-0.5 / sigma2 * x ** 2)
    
    if norm == 'flux':
        phi_x = phi_x / (1.0/np.sqrt(2))
    elif norm=='peak':
        phi_x = phi_x/ np.sum(np.power(phi_x,2))
    elif norm == 'avg': 
        phi_x =phi_x / phi_x.sum()
    elif norm == 'snr': 
        phi_x =phi_x / np.sqrt(np.sum(np.power(phi_x,2)))
    return phi_x


def filter1d(data, sigma, norm = 'avg',axis=-1, output=None, mode="reflect", cval=0.0, truncate=4.0,):
    """copy of scipy.ndimage.gaussian_filter1d with modified gaussian kernel normalizations"""
    sd = float(sigma)
    # make the radius of the filter equal to truncate standard deviations
    lw = int(truncate * sd + 0.5)
    # Since we are calling correlate, not convolve, revert the kernel
    weights = gaussian_kernel(sigma, lw, norm=norm)[::-1]
    return correlate1d(data, weights, axis, output, mode, cval, 0)


def get_spec_err(im, w=None, ra=None, dec=None, pbim=None, pbspec=None): 
    """
    Get spectrum and error at given ra, dec from image and wcs
    
    Parameters
    ----------

    im (array): image cube (y,x,chan), typically assumed to be primary-beam corrected.
    w (WCS): WCS object corresponding to image cube, optional
    ra (float): right ascension in degrees, or pixel x if w is None
    dec (float): declination in degrees, or pixel y if w is None
    pbim (array): primary beam image cube (y,x) or (y,x,chan). If None, assumes im is not pb corrected.
    pbspec (array): primary beam spectrum (chan), or None. If None, no pb spec correction is done.

    Returns
    -------
    spectrum (array): spectrum at given ra, dec
    err (array): error in spectrum at given ra, dec
    """
    if pbim is None: pbim = 1.0
    elif pbim.ndim < 3:
        pbim = pbim[..., np.newaxis]
    im_uncor = im * pbim # uncorrect for primary beam (consistent noise characteristics)
    if w is not None: px,py = w.world_to_pixel(SC(ra,dec,unit='deg'))
    else: px,py = ra, dec
    spectrum = im[int(np.round(py,0)),int(np.round(px,0)),:]
    err = get_channel_err(im_uncor, pbspec=pbspec)
    return spectrum, err

def id_spectrum_peak(spectrum):
    try:
        peak = np.nanargmax(spectrum) #this may not be the best peak ID design
    except ValueError:
        peak = 0
    return peak


def filter_and_id_peaks(coord, im, pbspec, pbim, freqs, w, sig, dv, rms_r=10, norm='avg'):
    """Perform Gaussian filtering of cube and identify peak"""
    if not type(w) == wcs.WCS:
        warnings.filterwarnings("ignore", category=FITSFixedWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        w = wcs.WCS(w)
    if not type(coord) == SC:
        coord=SC(*coord,unit='deg')
    uncor = im*pbim
    kernel_kms = sig2fwhm_kms(sig, dv) #FWHM of kernel in km/s
    smoothed = filter1d(uncor, sigma = sig, axis=2, norm=norm)
    px,py = w.world_to_pixel(coord)
    spectrum = smoothed[int(np.round(py,0)),int(np.round(px,0)),:]
    peak = id_spectrum_peak(spectrum)
    updated_coord, (px, py), peakflux, rms, snr, _ = flux_snr_2D(smoothed[:,:,peak]*dv, w, coord=coord, 
                                                               rms_r = rms_r, search_r = 0.5)
    #measure flux and rms
    rms /= pbspec
    peakflux /= pbspec[peak]

    #info dictionary to return
    peak_info = {}
    peak_info['updated_coord'] = updated_coord
    peak_info['peak_ind'] = peak
    peak_info['freq'] = freqs[peak]
    peak_info['flux'] = peakflux
    peak_info['flux_err'] = peakflux/snr
    peak_info['snr'] = snr
    peak_info['filter_size']= kernel_kms

    return peak_info
    

def match_filter_pipeline(run_name, source, coord, cub, pbspec, pbim, rms_r=10, velres = [100, 300, 600, 900], verbose=False, 
                   parallel=False):
    """Copy a spectral cube and smooth it with varying gaussian filters, testing for the highest significance peak
    
    Parameters
    ----------
    
    run_name (str): name of the run (for logging purposes)
    source (str): name of the source (for logging purposes)
    coord (SkyCoord): sky coordinate of source
    cub (Cube): interferopy Cube object
    pbspec (array): primary beam spectrum (chan)               
    pbim (array): primary beam image cube (y,x) or (y,x,chan)
    rms_r (float): radius in arcseconds outside of which to compute rms for S/N calculation
    velres (list): list of velocity resolutions (FWHM in km/s) for matched filtering kernels
    verbose (bool): whether to print out progress information
    parallel (bool): whether to parallelize over different velocity resolutions

    Returns
    -------
    bestresult (dict): dictionary with results from matched filtering, chosen as the highest S/N among different kernel sizes
    """
    if verbose: print('\n**********************************************************')
    if verbose: print(f'Running matched filter for source {source}...')
    if verbose: print('**********************************************************\n')

    dv=cub.deltavel()
    freqs = np.array(cub.freqs)
    if verbose: print('Making smoothed cubes...')
    #make copies of relevant portion of cube for different smoothings
    copies, pbcut, w = copy_sliced_cube(cub, coord, pbim, rms_r=rms_r, ncopies=len(velres))
    if verbose: print('Done. \n\nIdentifying peaks...')
    if parallel:
        ncores = min(os.cpu_count()-1, len(velres))
        pbcut_arr = np.array(pbcut.data)
        hdr = w.to_header()
        pbspec_arr = np.array(pbspec.data)
        with parallel_backend("loky"): #parallelize for speed
            results = Parallel(n_jobs=ncores)(
                delayed(filter_and_id_peaks)(run_name, coord, copies[i], pbspec_arr, pbcut_arr, freqs,
                                         hdr, fwhm_kms2sig(vr, dv), dv,  rms_r=rms_r,verbose=verbose,norm='avg') 
                                         for i, vr in enumerate(velres))
    else:
        results=[]
        for i, vr in enumerate(velres):
            sig = fwhm_kms2sig(vr, dv)
            results.append(filter_and_id_peaks(run_name, coord, copies[i], pbspec, pbcut.data, freqs,
                                               w, sig, dv, rms_r=rms_r, verbose=verbose,norm='avg'))
    if verbose: print("Done.\n")
    snrs = np.array([r['snr'] for r in results])
    try: 
        bestresult = results[np.nanargmax(snrs)] # choose highest s/n
    except ValueError: return None
    if verbose: print(f"Best S/N for cube: S/N = {bestresult['snr']}")
    if verbose: print(f"Best kernel size = {bestresult['filter_size']}")
    if verbose: print('Done.\n')
    
    return bestresult


