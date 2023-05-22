'''
This module shall provide capability to :
1) <CREATE MAP> Create an EUV Carrington map assimilating AIA 193 data 
over one carrington rotation centered at an input datetime, save as a 
fits file
2) <Extract Observed CH> Load a pre-computed fits file and apply EZSEG 
3) <DO CH SCORE> Read in an "observed CH" map and "modeled CH" map, 
compute precision, recall and f-score
'''

import astropy.units as u

def create_euv_map(center_date,euv_obs_cadence=1*u.day,
                   save_dir='./'
                   ) :
    '''
    Given `center_date`:`datetime.datetime` download a Carrington 
    rotation of EUV 193 Angstrom data centered around that date at
    an observational cadence of `euv_obs_cadence` (validate against
    some possible easy values). For each map, reproject into the 
    carrington frame of reference, and blend together with a 60 deg
    carrington longitude gaussian weighting to obtain a full sun EUV
    map. Save as fits file in `save_dir`:`str`
    '''
    pass

def extract_obs_ch(euv_map,save_dir='./') :
    '''
    Given `euv_map` fits file, read in, apply the EZSEG module, to 
    extract coronal hole contours. Convert contours to open and 
    closed pixels, create sunpy.map, save as fits file in `save_dir` 
    '''
    pass

def do_ch_score(ch_map_model, ch_map_obs,auto_interp=False) :
    '''
    Given model coronal hole binary map `ch_map_model` and observed 
    coronal hole binary map `ch_map_obs`, read both in, validate the
    dimensions are the same, if `auto_interp` automatically try to
    resample, if not `auto_interp` raise error if dimensions don't 
    match. Once validation is complete, compute precision, recall and
    f-score on the two binary maps.
    '''
    pass