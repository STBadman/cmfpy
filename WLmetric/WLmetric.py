'''
This module shall provide capability to :
1) <CREATE MAP> Create a WL Carrington map assimilating SOHO/LASCO data over one carrington rotation, save as fits file
2) <EXTRACT OBSERVED NL> Load a pre-computed fits file and apply Poirier routines to extract contour 
3) <DO WL SCORE> Read in an "observed NL" map and "modeled NL" map, compute normalized distance between curves
'''

import astropy.units as u

def create_WL_map(center_date, coronagraph_altitude=3*u.R_sun) :
    '''
    Given `center_date`:`datetime.datetime`, download SOHO/LASCO
    coronagraph images for one Carrington rotation centered on that
    date, extract white light intensity above the limbs at the 
    specified altitude, and assemble into a Carrington map. Save as
    fits file.
    '''
    pass

def extract_NL_obs(wl_map,save_dir='./') :
    '''
    Given a precomputed input White light carrington map (`wl_map`),
    extract the streamer maximum brightness (SMB) line, and the 
    half brightness contour as a function of longitude. (Following 
    Poirier+2021, Badman+2022). Save in `save_dir` 
    '''
    pass

def compute_WL_score(model_nl_map,smb_obs) :
    '''
    Given `model_nl_map` (user provided) and a precomputed `smb_obs`
    dataset describing the coronagraph-observed neutral line, extract
    the 1d model neutral line, and apply the Poirier+ method to compute
    the average angular distance between the two 1d curves weighted by
    the local streamer belt thickness. 
    '''
    pass