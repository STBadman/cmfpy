'''
Multipurpose helper functions used throughout this repository.
'''
import astropy.constants as const
from astropy.coordinates import SkyCoord
import astropy.units as u
import astrospice
import datetime
import numpy as np
import pandas as pd
from pfsspy import utils as pfss_utils
import sunpy.map
import sys
from bs4 import BeautifulSoup
import requests
from typing import get_type_hints

# Load in PSP/SolO Spice Kernels (download happens automatically)
kernels = []
SC_ALL = {"psp":"SOLAR PROBE PLUS",
          "solar orbiter":"SOLO",
          "stereo-a":"STEREO AHEAD", 
          "stereo-b":"STEREO BEHIND"}
SUPPORTED_BODIES = {**SC_ALL,"earth":"EARTH","L1":"EARTH"}
coverage_dict = {}
for sc in SC_ALL.keys()  :
    try : 
        k_add = astrospice.registry.get_kernels(sc,'predict')
        kernels.append(k_add)
    except : 
        try : 
            k_add = astrospice.registry.get_kernels(sc,'recon')
            kernels.append(k_add)
        except : sys.stdout.write(f"No kernels located for {sc}")
    coverage_dict[sc] = [k.coverage(k.bodies[0].name) for k in k_add]
COVERAGE_LIMITS = {}
for sc, coverages in coverage_dict.items() :
    tmin = np.nanmin([c[0] for c in coverages])
    tmax = np.nanmax([c[1] for c in coverages])
    COVERAGE_LIMITS[sc] = [tmin,tmax]


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

def create_carrington_trajectory(datetime_array,body,obstime_ref=None) :

    if type(datetime_array) == datetime.datetime : datetime_array = [datetime_array]

    ### Error handle if requested body is known by astrospice
    assert body in SUPPORTED_BODIES.keys(), f"body {body} not in {SUPPORTED_BODIES.keys()}"

    ### Error handle if requested daterange is out of range of spice coverage
    if body in ["L1","earth"] : pass
    else : 
        assert (datetime_array[0] > COVERAGE_LIMITS[body][0].datetime) & (datetime_array[-1] < COVERAGE_LIMITS[body][-1].datetime), \
            f"Requested timerange outside of SPICE coverage for body '{body}' : {[c.datetime for c in COVERAGE_LIMITS[body]]}"

    ### Create SkyCoord for PSP in the inertial (J2000) frame
    trajectory_inertial = astrospice.generate_coords(
    SUPPORTED_BODIES.get(body), datetime_array
    )

    ### Transform to solar co-rotating frame (SLOW)
    trajectory_carrington = trajectory_inertial.transform_to(
        sunpy.coordinates.HeliographicCarrington(observer="self")
    )

    if body=="L1" : 
        L1_correction = 1.0- ((const.M_earth.value/
                        (const.M_earth.value
                         +const.M_sun.value))/3)**(1/3)
    else : L1_correction=1.0

    if obstime_ref is not None :
        trajectory_carrington = carr2SkyCoord(
            trajectory_carrington.lon,
            trajectory_carrington.lat,
            trajectory_carrington.radius*L1_correction,
            obstime=obstime_ref
        )
    else : 
        trajectory_carrington = carr2SkyCoord(
            trajectory_carrington.lon,
            trajectory_carrington.lat,
            trajectory_carrington.radius*L1_correction,
            obstime=datetime_array
        )


    return trajectory_carrington


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

def carr2SkyCoord(lon,lat,radius,obstime) :
    """
    'Cast Carrington Coordinates to astropy.coordinates.SkyCoord'
    Given a set of heliographic coordinates and an observation time,  return
    an `astropy.coordinates.SkyCoord` encoding that information.
    Useful for annotating `sunpy.map.Map` with additional data.
    """
    return SkyCoord(lon=lon,lat=lat,radius=radius,
                    frame = sunpy.coordinates.HeliographicCarrington(
                        observer="Earth",obstime=obstime
                        ),
                    representation_type="spherical"
                   )

def datetime2unix(dt_arr) :
    """Convert 1D array of `datetime.datetime` to unix timestamps"""
    return np.array([dt.timestamp() for dt in dt_arr])

def unix2datetime(ut_arr) : 
    """Convert 1D array of unix timestamps (float) to `datetime.datetime`"""
    return np.array([datetime.datetime.utcfromtimestamp(ut) for ut in ut_arr])

@u.quantity_input
def delta_long(r:u.R_sun,
               r_inner=2.5*u.R_sun,
               vsw=360.*u.km/u.s,
               omega_sun=14.713*u.deg/u.d,
               ):
    """ 
    Ballistic longitudinal shift of a Parker spiral connecting two
    points at radius r and r_inner, for a solar wind speed vsw. Solar
    rotation rate is also tunable
    """
    return (omega_sun * (r - r_inner) / vsw).to("deg")

def ballistically_project(skycoord,r_inner = 2.5*u.R_sun, vr_arr=None) :
    """
    Given a `SkyCoord` of a spacecraft trajectory in the Carrington frame,
    with `representation_type="spherical"`, and optionally an array of
    measured solar wind speeds at the same time intervals of the trajectory,
    return a SkyCoord for the trajectory ballistically projected down to 
    `r_inner` via a Parker spiral of the appropriate curvature. When `vr_arr`
    is not supplied, assumes wind speed is everywhere 360 km/s
    """
    if vr_arr is None : vr_arr = np.ones(len(skycoord))*360*u.km/u.s
    lons_shifted = skycoord.lon + delta_long(skycoord.radius,
                                             r_inner=r_inner,
                                             vsw=vr_arr
                                            )
    return SkyCoord(
        lon = lons_shifted, 
        lat = skycoord.lat,
        radius = r_inner * np.ones(len(skycoord)),
        representation_type="spherical",
        frame = skycoord.frame
    )

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

def parse_map(data:np.ndarray|list):
    type_check(locals(),parse_map)

    sinlat = np.linspace(-1,1,np.shape(data)[0])
    long = np.linspace(0,360,np.shape(data)[1])

    body = np.column_stack((sinlat, data)).astype(str)
    ln1 = [str(),]
    ln1 = np.hstack((ln1,long.astype(str)))
    
    return np.vstack((ln1,body))