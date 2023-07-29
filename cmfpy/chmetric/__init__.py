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
import cmfpy.chmetric.chmap as chmap
import os
import numpy as np
from pfsspy import utils as pfss_utils
from sunpy.coordinates import sun
from sunpy.net import Fido, attrs as a
import sunpy.map
import sys
import cmfpy.utils as utils

def most_prob_val_log2d(data) :
    # Assumes data vals are within range 0.1-1000
    # Most probable value in 2d data array. Takes log
    # first
    counts,edges = np.histogram(np.log10(data),
                      bins=np.linspace(0,3,100)
                      )
    bin_centers = (edges[1:]+edges[:-1])/2
    peak = bin_centers[counts==np.nanmax(counts)][0]
    return peak   

import h5py
def compute_normalization_factor() :
   ### There's hardcoded number in the next function used to
   ### to rescale the pixel values in the Carrington map to match
   ### that used in Badman+2022. This function uses the example
   ### EUV carrington map located in `example_data` to produce 
   ### this normalization factor.
    euv_CH_path = f"./example_data/carr_maps.2210.final.h5"
    f = h5py.File(euv_CH_path, 'r')

    # Parse file into dictionary for simpler format
    names = [k for k in f.keys()]
    maps = [m[0] for m in f.values()]
    map_dict = dict(zip(names[0:5],[m for m in maps[0:5]]))
    f.close()

    map193 = map_dict["map_0193"][0]
    return most_prob_val_log2d(map193)

def create_euv_map(center_date,
                   euv_obs_cadence=1*u.day,
                   gaussian_filter_width = 30*u.deg,
                   days_around = 14, # number of days plus/minus the center date to create the map
                   save_dir=os.path.join(f'{__path__[0]}','data'),
                   replace=False,
                   wvln = 193*u.angstrom
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
    savepath = os.path.join(f"{save_dir}",f"{center_date.strftime('%Y-%m-%d')}_{int(wvln.value):04d}.fits")

    if not os.path.exists(savepath) or replace :

        ## Use sunpy Fido to search for AIA 193 data over a carrinton rotation
        ### If the downloads succeed once, they are archived on your computer in
        ### ~/sunpy/data and a database entry is created. This step then becomes 
        ### much quicker on the next run.
        sys.stdout.write(f"Searching for input EUV maps")
        res=Fido.search(
            a.Time(center_date-datetime.timedelta(days=days_around), 
                center_date+datetime.timedelta(days=days_around)
            ), 
            a.Instrument.aia,
            a.Wavelength(wvln), 
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
        carrington_maps, datetime_list = [], []
        sys.stdout.write(f"Reprojecting {len(carrington_rotation)} Maps: \n")
        for ii,m in enumerate(carrington_rotation) :
            sys.stdout.write(f"{ii+1:02d}/{len(carrington_rotation)}\r")
            header =  sunpy.map.make_heliographic_header(
                m.date, m.observer_coordinate,shape_out, frame='carrington'
                )
            carrington_maps.append(m.reproject_to(header))
            datetime_list.append(datetime.datetime.strptime(m.meta['date-obs'], '%Y-%m-%dT%H:%M:%S.%f'))
        
        ## Combine maps together with gaussian weighting

        ### Make header for combined map (use central map as reference)  
        closest_datetime = min(datetime_list, key=lambda x: abs(x - center_date))
        ref_map = carrington_rotation[datetime_list.index(closest_datetime)]
        ref_coord = ref_map.observer_coordinate
        ref_header = sunpy.map.make_heliographic_header(
            ref_map.date, ref_coord, shape_out, frame="carrington"
        )

        ### Compute a gaussian weight for each pixel in each map.
        gaussian_weights = [
        np.exp(-((sunpy.map.all_coordinates_from_map(m).lon.to("deg").value 
                -sun.L0(m.date).to("deg").value + 180) % 360 - 180)**2
            /(2*gaussian_filter_width.to("deg").value**2)
            ) 
        for m in carrington_maps
        ]

        ### Average all maps together, rescale data to match 
        # maps from Badman+2022
        combined_map_data = np.nanmean([
                m.data*w for m,w in 
                zip(carrington_maps,gaussian_weights)
                ],
                axis=0)
        peak_val_raw = most_prob_val_log2d(combined_map_data)
        #1.742 is the output of chmetric.chmetric.compute_normalization_factor()
        ratio = 10**1.742 / 10**peak_val_raw
        combined_map_data /= ratio
        
        combined_map_gaussian_weights = sunpy.map.Map(
            combined_map_data,ref_header)

        ### Align LH edge with Carrington 0 for consistency
        combined_map_gaussian_weights_roll = pfss_utils.roll_map(
            combined_map_gaussian_weights)

        ## Save output combined map as fits file
        combined_map_gaussian_weights_roll.save(savepath,
                                                overwrite=replace)  

    ## Return output map filename
    return savepath

def extract_obs_ch(euv_map_path,
                   replace=False,
                   save_dir = os.path.join(f'{__path__[0]}','data'),
                   ezseg_version="fortran", # will add fortran wrapper option
                   ezseg_params = {"thresh1":10, ## Seed threshold
                                   "thresh2":75, ## Growing Threshhold
                                   "nc":7, ## at least 7 consecutive pixels to declare coronal hole area is connected
                                   "iters":100 # Do maximum 100 iterations
                                   }
                   ) :
    '''
    Given `euv_map` fits file, read in, apply the EZSEG module, to 
    extract coronal hole contours. Convert contours to open and 
    closed pixels, create sunpy.map, save as fits file in `save_dir` 
    '''
         
    # python version from ezseg.py
    if ezseg_version == "python":
        savepath = os.path.join(f"{save_dir}",
                                f"{os.path.basename(euv_map_path)[:-5]}_ch_extracted_python.fits"
        ) 
        ## Run only if file does not already exist or if `replace == True`
        if not os.path.exists(savepath) or replace :
            euv_map = sunpy.map.Map(euv_map_path) 
            euvmap_array= np.log10(euv_map.data)
            valid_data = ~np.isnan(euv_map.data)

            ## Python version via D. H. Brooks
            segmented_array = chmap.ezsegpy(euvmap_array, ## Data to extract contours from
                                                    valid_data, ## Valid pixels
                                                    euvmap_array.shape[0], ## x-dimension of array
                                                    euvmap_array.shape[1], ## y-dimension of array
                                                    ezseg_params["thresh1"],
                                                    ezseg_params["thresh2"],
                                                    ezseg_params["nc"],
                                                    ezseg_params["iters"]
                                                )
            
            ## Cast to sunpy.map and save as fits file
            ch_map_obs = sunpy.map.Map(np.invert(segmented_array).astype(float),euv_map.meta)
            ch_map_obs.save(savepath, overwrite=replace)

    if ezseg_version == "fortran":
        savepath = os.path.join(f"{save_dir}",
                                "{os.path.basename(euv_map_path)[:-5]}_ch_extracted_fortran.fits"
                                )

        ## Run only if file does not already exist or if `replace == True`
        if not os.path.exists(savepath) or replace :
            euv_map = sunpy.map.Map(euv_map_path)
            data = euv_map.data
            use_indices = np.logical_and(data > 2., data!=np.nan)
            use_chd = use_indices.astype(int)
            use_chd = np.where(use_chd == 1, use_chd, np.nan)
            nx = euv_map.meta['naxis2']
            ny = euv_map.meta['naxis1']

            # fortran chd algorithm
            np.seterr(divide='ignore')
            ezseg_output, iters_used = chmap.ezseg(np.log10(data), use_chd, nt=nx, np=ny, 
                                                         thresh1=ezseg_params["thresh1"],
                                                         thresh2=ezseg_params["thresh2"],
                                                         nc=ezseg_params["nc"],
                                                         iters=ezseg_params["iters"])
            chd_result = np.logical_and(ezseg_output == 0, use_chd == 1)
            chd_result = chd_result.astype(int)

            # create CHD map
            ch_map_obs = sunpy.map.Map(chd_result.astype(int), euv_map.meta)                
            ch_map_obs.save(savepath, overwrite=replace)
        
    return savepath


def do_ch_score(dt_model, chmap_model, chmap_obs,auto_interp=False) :
    '''
    Given model coronal hole binary map `ch_map_model` and observed 
    coronal hole binary map `ch_map_obs`, read both in, validate the
    dimensions and y-axis projection are the same, if `auto_interp` 
    automatically try to resample, if not `auto_interp` raise error 
    if dimensions don't match. Once validation is complete, compute 
    precision, recall and f-score on the two binary maps.
    '''
    chmap_model = utils.csv2map(chmap_model, dt_model)
    chmap_obs = sunpy.map.Map(chmap_obs)
    if auto_interp :
        chmap_obs = chmap_obs.reproject_to(chmap_model.wcs)
    
    ## Check maps are the same shape and projection, to safely to 
    # binary computation
    assert (chmap_model.data.shape == chmap_obs.data.shape), \
        "Input maps shapes do not match, setting auto_interp=True" \
        + " will attempt to reproject the observed map to the shape" \
        + " of the modeled map."
    assert (chmap_model.meta['ctype2'] == chmap_obs.meta['ctype2']), \
        "Input maps projections do not match, setting auto_interp=True" \
        + " will attempt to reproject the observed map to the shape" \
        + " of the modeled map."
    
    ## Do Binary Computation
    model_bool = chmap_model.data.astype(bool)
    obs_bool = chmap_obs.data.astype(bool)

    ## Recall pixel value 1 = magnetically open/coronal hole, 
    # 0 = magnetically closed, not coronal hole
    tp = np.sum(model_bool & obs_bool) # True positive : pixel value 1 in model and obs
    fp = np.sum(model_bool & ~obs_bool) # False Positive : pixel value 1 in model and 0 in obs
    fn = np.sum(~model_bool & obs_bool) # False Negative : pixel value 0 in model and 1 in obs

    p = tp/(tp+fp) # precision : fraction of predicted open pixels which are actually open
    r = tp/(tp+fn) # recall : fraction of observed open pixels which are correctly predicted 
    f = 2*p*r/(p+r) # f-score : harmonic mean of p and r

    return p, r, f