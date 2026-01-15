import warnings
from glob import glob, iglob
import re
import numpy as np
from astropy.io import fits
from astropy.wcs import wcs
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70.0, Om0=0.30)

from .array_utils import load_cube

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
    if path is None: path = cub.filename
    try:
        spw = int(path.split('spw')[1].split('.')[0])
    except ValueError:
        print(f'path ({path}) had incorrect format for IDing spectral window')
    if cub is None: cub = load_cube(path)
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



def get_sources_folders(path_to_data, source_name_convention='ALMA_'):
    '''
    Get list of source names and folders for a set of data organized with the ALMA convention.

    Parameters
    ----------
    path_to_data (str): path to the root of the data folder (e.g., "2024.00042.S/")

    source_name_convention (str): root form of source names in ALMA archive, e.g., "ALMA_", "COSMOS_", 
        that will have either numbers or other symbols appended.

    Returns
    -------
    sourceslist: list of source names

    folder_labels: folders within 'path_to_data'
    '''
    folder_labels = glob('*',root_dir=path_to_data)
    sources={}
    for f in folder_labels:
        dir = path_to_data+f+'/' # change to get directory
        files = glob(dir+'**/*cube.I.pbcor.fits',recursive=True)
        sources[f] = []
        for filename in files:
            try:
                sources[f].append(source_name_convention+filename.split(source_name_convention)[1].split('_')[0]) #just save source name
            except: pass
        sources[f]=list(set(sources[f]))
    sourceslist = []
    for l in sources.values(): sourceslist+=l
    sourceslist = np.array(list(set(sourceslist)))
    sourceslist =sourceslist[np.argsort([int(re.findall(r'\d+', s)[0]) for s in sourceslist])]
    return sourceslist


def get_paths_for_source(sourcename,  path_to_search, coord=None):
    '''Get paths (relative to CWD) for a given sourcename and path to data. If coordinate provided, check if source coordinate is contained in file.
    
    Parameters
    -----------
    sourcename (str): name of source (e.g., returned by get_sources_folders())

    coord (astropy.coordinates.SkyCoord, optional): sky coordinate of source

    path_to_search: path to the root directory of data, in which to search for files
    
    Returns
    -------
    paths: list of paths to cube files for source
    '''
    paths=[]
    d = sourcename.split('CDFS_')[1] # number of source, assume sourcename is of form 'CDFS_#'
    initpaths = iglob(path_to_search+'**/*'+d+'*cube.I.pbcor.fits',recursive=True)
    if coord is not None:
        for p in initpaths:
            hdr = fits.getheader(p)
            w = wcs.WCS(hdr,naxis=2)
            if w.naxis > 2: w=w.dropaxis(2)
            if coord.contained_by(w): paths.append(p)
    else: paths = list(initpaths)
    return paths

def path_from_spw_band(source, path_to_data, spw, band):
    '''
    get paths that correspond to a given spw and band for a given source
    
    Parameters
    ----------
    source (str): name of source

    path_to_data (str): path to root folder of data

    spw (int): number of spectral window

    band (int): ALMA band number (3,4,or 6)

    
    Returns
    -------
    corr_paths (list): list of paths with correct spw and band    
    '''
    paths = get_paths_for_source(source, path_to_data)
    corr_paths = []
    for p in paths:
        spwi, bandi,_ = spw_band_from_path(p)
        if spwi==spw and bandi==band:
            corr_paths.append(p)
    return corr_paths

