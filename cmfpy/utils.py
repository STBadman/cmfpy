import astropy.units as u
import datetime
import numpy as np
import pandas as pd
from pfsspy import utils as pfss_utils
import sunpy.map
from bs4 import BeautifulSoup
import requests
from typing import get_type_hints
from scipy.interpolate import NearestNDInterpolator, CloughTocher2DInterpolator, LinearNDInterpolator
from astropy.convolution import interpolate_replace_nans, convolve, Tophat2DKernel

def npy2map(npypath,datetime_npy) :
    """
    Helper function which expects CSV files prepared in the same format as
    provided in Badman+2022 and the example files in ./example_model_data
    Reads in CSV files, assumeds sinlat binning, assumes first row and first
    column are coordinates, generates header, creates sunpy.map.Map
    """
    
    data = np.load(npypath,allow_pickle=True)[1:,1:]

    header = pfss_utils.carr_cea_wcs_header(datetime_npy,
                                            data.T.shape,
                                            map_center_longitude=180*u.deg
                                            )

    return sunpy.map.Map(data,header)

def type_check(args,func):
    type_hints = get_type_hints(func)

    for arg in type_hints:
        hint_type = type_hints[arg]
        arg_val = args[arg]
        arg_type = type(arg_val)

        if not isinstance(arg_val,hint_type):
            try: msg = f"'{arg}' is of type {arg_type.__name__}; expected {hint_type.__name__}"
            except: msg = f"'{arg}' is of type {arg_type.__name__}; expected {hint_type}"
            raise TypeError(msg)

def listhtml(url:str, contains:str='', include_url:bool=True):
    type_check(locals(),listhtml)

    page = requests.get(url).text
    soup = BeautifulSoup(page, 'html.parser')

    out = [node.get('href') for node in soup.find_all('a')]
    if include_url: out = [f'{url}{f}' for f in out if contains in f]
    else: out = [f'{f}' for f in out if contains in f]
    return out

def parse_to_datetime(date:str, delim:str=':'):
    type_check(locals(),parse_to_datetime)

    year, month, day, time = np.array(date.split(delim),dtype=str)
    hour, minute = time[0:2], time[2:]
    date_datetime = datetime.datetime( int(year),int(month),int(day),int(hour),int(minute) )

    return date_datetime

def parse_from_datetime(date_datetime:datetime.datetime, delim:str=':'):
    
    year, month, day = f'{date_datetime.year}', f'{date_datetime.month:02}', f'{date_datetime.day:02}'
    hour, minute = f'{date_datetime.hour:02}', f'{date_datetime.minute:02}'

    date = f'{year}{delim}{month}{delim}{day}{delim}{hour}{minute}'

    return date

def datetime2unix(dt_arr) :
    """Convert 1D array of `datetime.datetime` to unix timestamps"""
    return np.array([dt.timestamp() for dt in dt_arr])

def unix2datetime(ut_arr) : 
    """Convert 1D array of unix timestamps (float) to `datetime.datetime`"""
    return np.array([datetime.datetime.utcfromtimestamp(ut) for ut in ut_arr])

def gen_dt_arr(dt_init,dt_final,cadence_days=1) :
    """
    'Generate Datetime Array'
    Get array of datetime.datetime from {dt_init} to {dt_final} every 
    {cadence_days} days
    """
    dt_list = []
    while dt_init < dt_final :
        dt_list.append(dt_init)
        dt_init += datetime.timedelta(days=cadence_days)
    return np.array(dt_list)

def replace_nans(data,interpolator=LinearNDInterpolator):
    # first remove the majority of nans from the body of the image
    mask = np.where(np.isfinite(data))
    interp = interpolator(np.transpose(mask), data[mask])
    data = interp(*np.indices(data.shape))

    # remove any nans on the edge of the image
    data = interpolate_replace_nans(data, Tophat2DKernel(1), convolve=convolve, boundary='extend')

    return data