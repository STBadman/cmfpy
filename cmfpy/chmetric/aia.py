### Imported from https://github.com/dstansby/solarsynoptic/blob/main/solarsynoptic/data/helpers.py

from datetime import datetime, timedelta
from pathlib import Path

from astropy.time import Time
from sunpy.net import Fido
from sunpy.net import attrs as a
from sunpy.time import parse_time


def start_of_day(dtime):
    if isinstance(dtime, Time):
        dtime = dtime.datetime
    # Get datetime at start of current day
    return datetime(dtime.year, dtime.month, dtime.day)


def start_of_day_map(dtime, *query):
    """
    Download the first map available on a given date, with the given Fido query.

    Parameters
    ----------
    dtime :
        Datetime.
    query :
        Query parameters.
    """
    dtime = parse_time(dtime).to_datetime()
    if dtime > datetime.now():
        raise RuntimeError("Can't fetch map in the future!")
    # Get datetime at start of current day
    dtime = start_of_day(dtime)

    query = (a.Time(dtime, dtime + timedelta(days=1), dtime), *query)
    result = Fido.search(*query)
    try:
        download_path = Fido.fetch(result[0, 0])[0]
    except IndexError:
        raise RuntimeError(f'No maps available for whole day on {dtime}')

    return Path(download_path)

### Imported from https://github.com/dstansby/solarsynoptic/blob/main/solarsynoptic/data/aia.py
import pathlib
from datetime import datetime, timedelta

import astropy.units as u
from sunpy import config as sunpy_config
from sunpy import log
from sunpy.map import Map
from sunpy.net import attrs as a
from sunpy.time import parse_time

#from . import helpers

__all__ = ['aia_start_of_day_map']

def aia_start_of_day_map(dtime, wlen, dl_path=None, mode="JSOC"):
    """
    Download and load the first AIA map available on a given date at a given
    wavelength.

    By default files are downloaded to the sunpy download directory.

    Parameters
    ----------
    dtime : astropy.time.Time
    wlen : astropy.units.Quantity
    dl_path : str, pathlib.Path
        Directory to search for files and download files to. If `None` defaults
        to the sunpy download directory.

    Returns
    -------
    sunpy.map.AIAMap

    Notes
    -----
    For dates more than 14 days in the future calibrated L0 maps are
    downloaded. For dates more recent than that a near-real-time file is
    downloaded.
    """
    dtime = parse_time(dtime).to_datetime()
    dtime = start_of_day(dtime)
    if dl_path is None:
        dl_path = sunpy_config.get('downloads', 'download_dir')
    dl_path = pathlib.Path(dl_path)

    # Download from JSOC if older than 14 days; otherwise directly
    # get an NRT map
    if (dtime < datetime.now() - timedelta(days=13)) and (mode != "JSOC") :
        map_path = _map_path(dtime, wlen, dl_path)
        if not map_path.exists():
            query = a.Instrument('AIA'), a.Wavelength(wlen)
            log.info(f'Downloading AIA {int(wlen.to_value(u.Angstrom))} map '
                     f'to {map_path}')
            dl_path = start_of_day_map(dtime, *query)
            dl_path = pathlib.Path(dl_path)
            dl_path.replace(map_path)
    else:
        map_path = _nrt_map_path(dtime, wlen, dl_path)
        if not map_path.exists():
            import parfive
            dl = parfive.Downloader(max_conn=1)
            wlen_int = int(wlen.to_value(u.Angstrom))
            url = (f"http://jsoc2.stanford.edu/data/aia/synoptic/"
                   f"{dtime.year}/{dtime.month:02}/{dtime.day:02}/"
                   f"H0000/AIA{dtime.year}{dtime.month:02}{dtime.day:02}_"
                   f"0000_0{wlen_int}.fits")
            dl.enqueue_file(url, filename=map_path)
            log.info(f'Downloading AIA {int(wlen.to_value(u.Angstrom))} '
                     f'map to {map_path}')
            res = dl.download()
            if len(res.errors):
                log.info(res.errors)
                raise RuntimeError('Download failed')
            dl_path = pathlib.Path(res[0])
            dl_path.replace(map_path)

    #smap = Map(map_path)
    return map_path#smap


def _map_path(dtime, wlen, directory):
    datestr = dtime.strftime('%Y%m%d')
    wlen = int(wlen.to_value(u.Angstrom))
    return directory / f'aia_{wlen}_{datestr}.fits'


def _nrt_map_path(dtime, wlen, directory):
    datestr = dtime.strftime('%Y%m%d')
    wlen = int(wlen.to_value(u.Angstrom))
    return directory / f'aia_{wlen}_{datestr}_nrt.fits'