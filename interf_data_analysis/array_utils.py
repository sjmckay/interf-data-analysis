import numpy as np
from astropy.cosmology import FlatLambdaCDM
import matplotlib.pyplot as plt
import warnings
from interferopy.cube import Cube
from io import StringIO
from contextlib import redirect_stdout
from astropy.nddata import Cutout2D
import astropy.units as u
cosmo = FlatLambdaCDM(H0=70.0, Om0=0.30) 

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

def find_map_peak(map, mask=None):
    '''find peak pixel in a 2D cutout. Returns pixel coords in row, col (y, x) format (i.e., numpy format)'''
    nmap = map.copy()
    if mask is not None:
        nmap[mask] = np.nan
        if np.all(np.isnan(nmap)):
            warnings.warn("All nan map for peak finding. Returning center...")
            return nmap.shape[0]//2, nmap.shape[1]//2
    return np.unravel_index(np.nanargmax(nmap,),nmap.shape)


def rebin(arr, factor=None,new_len=None,function='mean'):
    """Rebin 2D array arr to shape new_shape by averaging/summing. If the factor by which to rebin does not
    fit evenly into the old array, extra elements are discarded.
    
    Parameters
    ----------
    arr (array): array to rebin
    
    factor(int or float): optional factor (number of channels) by which to rebin
    
    new_len (int): optional new length of array (in case you don't want to input as a factor to downsize by)
    
    function (str): either 'mean', 'quad', or 'sum'. If mean or sum, applies those numpy functions. 
        If quad, sums in quadrature and divides by sqrt(factor) to get reduced error.
        
        
    Returns
    -------
    (array): rebinned array with function applied.
    """
    
    if function =='sum': func = np.nansum
    elif function == 'quad': 
        func = lambda x, axis: np.sqrt(np.nansum(x**2,axis=axis))/x.shape[axis]
    else: func = np.nanmean
    
    if factor is not None:
        shape = (len(arr)//factor,factor)   
        max_index = len(arr)-len(arr)%factor
        newarr = arr[:max_index].copy()
        return func(newarr.reshape(shape),axis=-1)
    if new_len is not None:
        shape = (new_len,len(arr)//new_len)   
        max_index = (len(arr)//new_len)*new_len
        newarr = arr[:max_index].copy()
        return func(newarr.reshape(shape), axis=-1)
    raise ValueError('Must input either factor or new_len to rebin function.')


def load_cube(path):
    f=StringIO()
    with redirect_stdout(f):
        return Cube(path)
    


def copy_sliced_cube(cub, coord, pbim, rms_r=10,ncopies=1):
    """make repeated copies of portion of cube, retaining a square of side s=2*rms_r arcseconds around the coordinate"""
    w = cub.wcs
    w2 = w.copy()
    if w2.naxis > 2: w2=w2.dropaxis(2)
    # cutout of primary beam image
    pbcut = Cutout2D(pbim, position=coord, size=2*rms_r*u.arcsec, wcs=w2, copy=True, mode='trim')
    newwcs = pbcut.wcs
    slices = pbcut.slices_original
    cut = cub.im[slices[1],slices[0],:]
    cut=np.swapaxes(cut,0,1)
    # make copies of cube cutout if desired
    if ncopies == 1: return cut, pbcut, newwcs
    copies=[np.copy(cut) for _ in range(ncopies)]
    return copies, pbcut.data[:,:,np.newaxis], newwcs