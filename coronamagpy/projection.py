'''
Multipurpose helper functions used throughout this repository.
'''
import astropy.constants as const
from astropy.coordinates import SkyCoord
import astropy.units as u
import astrospice
import datetime
import numpy as np
import sunpy.map
import sys

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