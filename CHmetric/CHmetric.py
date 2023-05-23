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
import datetime
import os
import numpy as np
from sunpy.coordinates.sun import L0
from sunpy.net import Fido, attrs as a
import sunpy.map
import sys

def create_euv_map(center_date,
                   euv_obs_cadence=1*u.day,
                   save_dir='./CHmetric/data/'
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

    ## First, check if map centered on center_date has already been created 
    ## if it has, jump straight to the end.
    savepath = f"{save_dir}/{center_date.strftime('%Y-%m-%d')}.fits"
    if not os.path.exists(savepath) :

        ## Use sunpy Fido to search for AIA 193 data over a carrinton rotation
        ### If the downloads succeed once, they are archived on your computer in
        ### ~/sunpy/data and a database entry is created. This step then becomes 
        ### much quicker on the next run.
        sys.stdout.write(f"Searching for input EUV maps")
        res=Fido.search(
            a.Time(center_date-datetime.timedelta(days=14), 
                center_date+datetime.timedelta(days=14)
            ), 
            a.Instrument.aia,
            a.Wavelength(193*u.angstrom), 
            a.Sample(euv_obs_cadence)
            )  
        ## Download, or return files if already exist
        downloaded_files = Fido.fetch(res)

        ## Read in downloaded data as `sunpy.map.Map` objects and downsample
        downsample_dims = [1024,1024] * u.pixel
        carrington_rotation = [sunpy.map.Map(m).resample(downsample_dims) 
                            for m in downloaded_files]
        
        ## Loop through input maps and reproject each one to the Carrington frame
        shape_out = (360, 720)
        carrington_maps = []
        sys.stdout.write(f"Reprojecting {len(carrington_rotation)} Maps: \n")
        for ii,m in enumerate(carrington_rotation) :
            sys.stdout.write(f"{ii+1:02d}/{len(carrington_rotation)}\r")
            header =  sunpy.map.make_heliographic_header(m.date, m.observer_coordinate, 
                                            shape_out, frame='carrington')
            carrington_maps.append(m.reproject_to(header))
        
        ## Combine maps together with gaussian weighting

        ### Make header for combined map (use central map as reference)
        ref_map = carrington_rotation[14]
        ref_date = ref_map.meta['date-obs']
        ref_coord = ref_map.observer_coordinate
        ref_header = sunpy.map.header_helper.make_heliographic_header(
            ref_map.date, ref_coord, shape_out, frame="carrington"
        )

        ### Compute a gaussian weight for each pixel in each map.
        gaussian_width_lon = 60
        gaussian_weights = [
        np.exp(-((sunpy.map.all_coordinates_from_map(m).lon.to("deg").value 
                -L0(m.date).to("deg").value + 180) % 360 - 180)**2
            /(2*gaussian_width_lon**2)
            ) 
        for m in carrington_maps
        ]

        ### Average all maps together
        combined_map_gaussian_weights = sunpy.map.Map(
            np.nanmean([
                m.data*w for m,w in 
                zip(carrington_maps,gaussian_weights)
                ],
                axis=0),ref_header)

        ## Save output combined map as fits file
        combined_map_gaussian_weights.save(savepath)  

    ## Return output map filename
    return savepath

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