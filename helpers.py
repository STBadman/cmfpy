'''
Multipurpose helper functions used throughout this repository.
'''

import astropy.units as u
import numpy as np
import pandas as pd
from pfsspy import utils as pfss_utils
import sunpy.map

def csv2map(csvpath,datetime_csv) :
    """
    Helper function which expects CSV files prepared in the same format as
    provided in Badman+2022 and the example files in ./example_model_data
    Reads in CSV files, assumeds sinlat binning, assumes first row and first
    column are coordinates, generates header, creates sunpy.map.Map
    """
    
    data = np.array(pd.read_csv(csvpath,index_col=0,header=0,dtype=float))[1:,1:]

    header = pfss_utils.carr_cea_wcs_header(datetime_csv,
                                            data.T.shape,
                                            map_center_longitude=180*u.deg
                                            )

    return sunpy.map.Map(data,header)