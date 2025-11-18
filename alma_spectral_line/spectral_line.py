import numpy as np
import pandas as pd
pd.set_option('display.max_rows', None)
from interferopy.cube import Cube
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

import io
import warnings
warnings.filterwarnings("ignore", category=FITSFixedWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
from contextlib import redirect_stdout

from sjmm_util.datatools import rebin, find_map_peak
from .utils import distance_array

def spw_band_from_path(path=None,cub=None):
    '''
    Get spectral window and ALMA band for a given path/existing cube.

    Parameters
    ----------
    path (str): path for which to get spw and band

    cub: Cube object from interferopy, used to limit reading in from disk

    Returns
    -------
    spw (int): spectral window number (usually one of 25,27,29,31)

    band (int): ALMA band (currently only supports bands 3,4,6)

    folder (str): folder (per science goal for ALMA) that is unique to that path (e.g., Xc252)--only
        used to distinguish sources later on
    '''
    f = io.StringIO()
    if path is None: path = cub.filename
    try:
        spw = int(path.split('spw')[1].split('.')[0])
    except ValueError:
        print(f'path ({path}) had incorrect format for IDing spectral window')
    if cub is None:
        with redirect_stdout(f):
            cub = Cube(path)
    freq = cub.reffreq
    if freq < 130 and freq > 80:
        band = 3
    elif freq < 180 and freq > 130:
        band = 4
    elif freq < 330 and freq > 180:
        band = 6
    else:
        raise ValueError(f'The reference frequency ({freq} GHz) is not within the expected bands.')

    try:
        folder = path.split('/science_goal.')[1].split('/')[0].split('_')[-1]
    except Exception as e:
        warnings.warn(f'Could not determine folder for path {path}, raised exception {e}')
        folder = ''
    return spw, band, folder
    

def copy_sliced_cube(cub, coord, pbim, rms_r=10,ncopies=1):
    """make repeated copies of portion of cube, retaining a square of side s=2*rms_r arcseconds around the coordinate"""
    w = cub.wcs
    w2 = w.copy()
    if w2.naxis > 2: w2=w2.dropaxis(2)
    pbcut = Cutout2D(pbim, position=coord, size=2*rms_r*u.arcsec, wcs=w2, copy=True, mode='trim')
    newwcs = pbcut.wcs
    slices = pbcut.slices_original
    cut = cub.im[slices[1],slices[0],:]
    cut=np.swapaxes(cut,0,1)
    if ncopies == 1: return cut, pbcut, newwcs
    copies=[np.copy(cut) for _ in range(ncopies)]
    return copies, pbcut.data[:,:,np.newaxis], newwcs

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


def compute_rms(map, exclude_r=None, mask=None, sigma=3.0, maxiters=5,copy=True):
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
    n_total = len(data)
    clipped = sigma_clip(data, sigma=sigma, maxiters=maxiters, stdfunc=np.nanstd)
    rms = np.nanstd(clipped.data[~clipped.mask])
    nsamples = len(clipped.data[~clipped.mask])
    return rms, nsamples
    
    
def flux_snr_2D(linemap, wcsi, coord=None, rms_r = 10, search_r = 1):
    '''get location and flux of peak in line map'''
    ppas = np.abs(1.0/wcsi.pixel_scale_matrix[0,0]/3600)
    center_cut = wcsi.world_to_pixel(coord)
    dist = distance_array(linemap,center_cut)
    mask = dist>search_r*ppas
    py,px = find_map_peak(linemap,mask=mask) 
    peakflux = linemap[py,px] # double check order of axes
    rmsmask = dist>rms_r*ppas
    rms,nsamples = compute_rms(linemap,exclude_r=2.0*ppas, mask = rmsmask)
    snr = peakflux/rms
    ra,dec = wcsi.all_pix2world(px,py,0) #pass (ra,dec) --> (ra,dec)
    newcoord = SC(ra, dec, unit='deg')

    return newcoord, (px,py), peakflux, rms, snr, nsamples 


def get_err(cub_pbcor, r=0.,ra=None,dec=None, pbim=None, pbspec=None): 
    """Determine rms error in cube by channel, similar to interferopy get_err function but with modifications"""
    if pbim is None: pbim = 1.0
    elif pbim.ndim < 3:
        pbim = pbim[..., np.newaxis]
    if pbspec is None: pbspec=1.0
    im_uncor = cub_pbcor.im * pbim
    mask = False
    bc_mask = np.broadcast_to(mask, im_uncor.shape)
    im_uncor[bc_mask]=np.nan
    rms = np.zeros(im_uncor.shape[2])
    for i in range(im_uncor.shape[2]):
        rms[i] = iftools.calcrms(im_uncor[:,:,i])
    err = rms/pbspec
    if r>0.:
        npix = np.pi*(r/cub_pbcor.pixsize)**2
        err *= np.sqrt(npix / cub_pbcor.beamvol)
    return err


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


def get_spec_err_ndarray(im_pbcor, w, ra,dec, pbim=None, pbspec=None): 
    "retrieve spectrum and err in ndarray (no WCS information)"
    if pbim is None: pbim = 1.0
    elif pbim.ndim < 3:
        pbim = pbim[..., np.newaxis]
    if pbspec is None: pbspec=1.0
    im_uncor = im_pbcor * pbim
    px,py = w.world_to_pixel(SC(ra,dec,unit='deg'))
    spectrum = im_pbcor[int(np.round(py,0)),int(np.round(px,0)),:]
    rms = np.zeros(im_uncor.shape[2])
    for i in range(im_uncor.shape[2]):
        rms[i] = iftools.calcrms(im_uncor[:,:,i]) #compute rms in each channel
    err = rms/pbspec
    return spectrum, err

def get_spectrum(cub, w=None, coord=None, r=0.5, pbim=None, pbspec=None):
    '''wrapper function for measuring a spectrum and err in a circular aperture, using interferopy.cube.Cube'''
    if coord is not None: ra,dec=coord.ra, coord.dec
    else: ra, dec = None, None
    if not hasattr(cub,'im'): #if not dealing with an interferopy.Cube
        if w is None:
            raise Exception('Either a Cube or a WCS object must be supplied.')
        spectrum, err = get_spec_err_ndarray(cub,w=w, ra=ra,dec=dec,pbim=pbim, pbspec=pbspec)
    else: 
        err  = get_err(cub,r=r,ra=ra,dec=dec,pbim=pbim, pbspec=pbspec)
        spectrum = cub.aperture_value(ra=ra, dec=dec, radius=r, calc_error=False)
    return spectrum, err


def filter_and_id_peaks(coord, im, pbspec, pbim, freqs, w, sig, dv, rms_r=10, norm='avg'):
    """Perform Gaussian filtering of cube and identify peak"""
    if not type(w) == wcs.WCS:
        warnings.filterwarnings("ignore", category=FITSFixedWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        w = wcs.WCS(w)
    if not type(coord) == SC:
        coord=SC(*coord,unit='deg')
    uncor = im*pbim
    kernel_kms = int(round(sig*dv*2.355)) #FWHM of kernel in km/s
    smoothed = filter1d(uncor, sigma = sig, axis=2, norm=norm)
    px,py = w.world_to_pixel(coord)
    spectrum = smoothed[int(np.round(py,0)),int(np.round(px,0)),:]
    try:
        peak = np.nanargmax(spectrum) #this may not be the best peak ID design
    except ValueError:
        peak = 0
    linefreemask = np.abs(np.arange(len(spectrum))-peak)>4*sig
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
    peak_info['cont']= cont

    return peak_info
    

def filtered_peaks(run_name, source, coord, cub, pbspec, pbim, rms_r=10, verbose=False, 
                   parallel=False,norm='avg'):
    """Copy a spectral cube and smooth it with varying gaussian filters, testing for the highest significance peak"""
    spw, band, folder = spw_band_from_path(cub.filename)
    if verbose: print('\n**********************************************************')
    if verbose: print(f'\nIdentifying peaks for source {source} in band {band}, spw {spw}...\n')

    path = cub.filename
    dv=cub.deltavel()
    velres = [200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 500, 
              525, 550, 575, 600, 650, 700, 750, 800, 900, 1000, 1200, 1500]
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
                                         hdr, vr/2.355/dv, dv,  rms_r=rms_r,verbose=verbose,norm=norm) 
                                         for i, vr in enumerate(velres)
        )
    else:
        results=[]
        for i, vr in enumerate(velres):
            sig = vr/dv/2.355
            results.append(filter_and_id_peaks(run_name, coord, copies[i], pbspec, pbcut.data, freqs,
                                               w, sig, dv, rms_r=rms_r, verbose=verbose,norm=norm))
    if verbose: print("Done.\n")
    snrs = np.array([r['snr'] for r in results])
    try: 
        bestresult = results[np.nanargmax(snrs)] # choose highest s/n
    except ValueError: return None
    if verbose: print(f"Best S/N for cube: S/N = {bestresult['snr']}")
    if verbose: print(f"Best kernel size = {bestresult['filter_size']}")
    if verbose: print('Done.\n')
    
    return bestresult

