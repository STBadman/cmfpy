# %%
import cmfpy.chmetric as chmetric
import cmfpy.wlmetric as wlmetric
import cmfpy.nlmetric as nlmetric
import cmfpy.io
import cmfpy.utils as utils
import cmfpy.projection as projection
 
import urllib.request
import os

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

import pfsspy
import sunpy.map
import astropy.coordinates
import astropy.units as u
from scipy.signal import find_peaks
import datetime

def parse_map(data:np.ndarray|list):
    utils.type_check(locals(),parse_map)

    sinlat = np.linspace(-1,1,np.shape(data)[0])
    long = np.linspace(0,360,np.shape(data)[1])

    body = np.column_stack((sinlat, data)).astype(str)
    ln1 = [str(),]
    ln1 = np.hstack((ln1,long.astype(str)))
    
    return np.vstack((ln1,body))

def get_magneto(date:str, delim:str=':', return_name:bool=False):
    utils.type_check(locals(),get_magneto)

    year, month, day, time = np.array(date.split(delim),dtype=str)
        
    datestr = f'{year}{month}{day}{time}'
    prefix = f'adapt_'
    dataname = f'{prefix}{datestr}.ftz.gz'
    datadir = f'{__path__[0]}/data/'

    if not os.path.exists(f'{datadir}'): os.makedirs(f'{datadir}')

    if not os.path.exists(f'{datadir}/{dataname}'):
        url = f'https://gong.nso.edu/adapt/maps/gong/{year}/'
        data_url = utils.listhtml(url=url,contains=f'{datestr}')[0]

        urllib.request.urlretrieve(f'{data_url}', f'{datadir}/{dataname}')

    if return_name: return dataname

def adapt2pfsspy(filepath:str, #must already exist on your computer
                 rss:int|float=2.5, # Source surface height
                 nr:int=60, # number of radial gridpoiints for model
                 realization:str="mean", #which slice of the adapt ensemble to choose
                 return_magnetogram:bool=False # switch to true for function to return the input magnetogram
                ):
    utils.type_check(locals(),adapt2pfsspy)

    # Load the FITS file into memory
    # ADAPT includes 12 "realizations" - model ensembles
    # pfsspy.utils.load_adapt is a specific function that knows
    # how to handle adapt maps
    adaptMapSequence = pfsspy.utils.load_adapt(filepath)
    # If realization = mean, just average them all together
    if realization == "mean" : 
        br_adapt_ = np.mean([m.data for m in adaptMapSequence],axis=0)
        adapt_map = sunpy.map.Map(br_adapt_,adaptMapSequence[0].meta)
    # If you enter an integer between 0 and 11, the corresponding
    # realization is selected
    elif isinstance(realization,int) : adapt_map = adaptMapSequence[realization]
    else : raise ValueError("realization should either be 'mean' or type int ") 
    
    # pfsspy requires that the y-axis be in sin(degrees) not degrees
    # pfsspy.utils.car_to_cea does this conversion
    adapt_map_strumfric = pfsspy.utils.car_to_cea(adapt_map)

    # Option to return the magnetogram
    if return_magnetogram : 
        return adapt_map_strumfric
    # Otherwise run the PFSS Model and return
    else :
        # ADAPT maps input are in Gauss, multiply by 1e5 to units of nT
        adapt_map_input = sunpy.map.Map(adapt_map_strumfric.data*1e5,
                                        adapt_map_strumfric.meta)
        peri_input = pfsspy.Input(adapt_map_input, nr, rss)
        peri_output = pfsspy.pfss(peri_input)
        return peri_output

def pfss2flines(pfsspy_output, # pfsspy output object
                nth:int=18,nph:int=36, # number of tracing grid points
                rect:np.ndarray|list|tuple=[-1,1,0,360], #sub-region of sun to trace (default is whole sun)
                trace_from_SS:bool=False, # if False : start trace from photosphere, 
                                          #if True, start tracing from source surface
                skycoord_in=None, # Use custom set of starting grid poitns
                max_steps:int=1000 # max steps tracer should take before giving up
                ) :
    utils.type_check(locals(),pfss2flines)

    # Tracing if grid
    if skycoord_in is None  :
        [latmin,latmax,lonmin,lonmax]=rect
        lons,lats = np.meshgrid(np.linspace(lonmin,lonmax,nph),
                                np.linspace(latmin,latmax,nth)
                                )
        if not trace_from_SS : alt = 1.0*u.R_sun # Trace up from photosphere
        else : alt = pfsspy_output.grid.rss*u.R_sun  # Trace down from ss
        alt = [alt]*len(lons.ravel())
        seeds = astropy.coordinates.SkyCoord(lons.ravel()*u.deg,
                               (np.arcsin(lats)*180/np.pi).ravel()*u.deg,
                               alt,
                               frame = pfsspy_output.coordinate_frame)
        
    # Tracing if custom set of points
    else : 
        skycoord_in.representation_type = "spherical"
        seeds = astropy.coordinates.SkyCoord(skycoord_in.lon,
                               skycoord_in.lat,
                               skycoord_in.radius,
                               frame = pfsspy_output.coordinate_frame)
        
    return pfsspy_output.trace(pfsspy.tracing.FortranTracer(max_steps=max_steps),seeds)
        
def pfss(date:str,
           delim:str=':',
           rss:int|float=2.5,
           return_name:bool=False
           ):
    utils.type_check(locals(),pfss)

    outdir = f'{__path__[0]}/out'
    datadir = f'{__path__[0]}/data'

    if not os.path.exists(f'{outdir}'): os.makedirs(f'{outdir}')

    magname = get_magneto(date,delim,return_name=True)

    ofl_name = f'ofl_{rss}_{magname[6:18]}.csv'
    nl_name = f'nl_{rss}_{magname[6:18]}.csv'

    if not os.path.exists(f'{outdir}/{ofl_name}') or not os.path.exists(f'{outdir}/{nl_name}'):

        pfss_model = adapt2pfsspy(f'{datadir}/{magname}',rss=rss)
        ss_model = pfss_model.source_surface_br

        ss = ss_model.data

        nlmap = np.where(ss > 0, 1, ss)
        nlmap = np.where(ss < 0, -1, nlmap)

        flines_highres = pfss2flines(pfss_model,nth=180,nph=360)
        topologies = flines_highres.polarities.reshape([180,360])


        oflmap = np.abs(topologies)


        ofl_out = parse_map(oflmap)
        nl_out = parse_map(nlmap)


        np.savetxt(f'{outdir}/{ofl_name}', ofl_out, fmt = '%s', delimiter=',')
        np.savetxt(f'{outdir}/{nl_name}', nl_out, fmt = '%s', delimiter=',')

    if return_name: return ofl_name, nl_name, rss

def thresholds(euvmap:sunpy.map.mapbase.GenericMap):
    N=20
    logdata = np.log10(euvmap.data.flatten())
    logdata = logdata[np.isfinite(logdata)]
    logdata_hist, edges = np.histogram(logdata,bins=N)
    peaks, _ = find_peaks(logdata_hist)
    try:
        thresh1 = (edges[peaks[0]] + edges[peaks[1]])/2
        thresh1 = (thresh1 + edges[peaks[1]])/2
    except: thresh1 = edges[peaks[0]]*0.9
    return thresh1, 2*thresh1

def find_close_magneto_date(date:str,
                          delim:str=':',
                          days_around:int=14
                          ):
    utils.type_check(locals(),find_close_magneto_date)

    year, month, day, time = np.array(date.split(delim),dtype=str)

    url = f'https://gong.nso.edu/adapt/maps/gong/{year}/'
    files = utils.listhtml(url, contains=year, include_url=False)


    target_day = 30*int(month) + int(day) + int(time)/2400

    day_deltas = []

    for f in files:
        datestr = f[22:22+2+2+4]

        test_month = int(datestr[:2])
        test_day = int(datestr[2:4])
        test_time = int(datestr[4:])
        test_day = 30*test_month + test_day + test_time/2400
        delta = np.abs(target_day-test_day)
        day_deltas.append(delta)

    if np.min(day_deltas) > days_around: raise RuntimeError('Closest date available for magnetogram data exceeds the limit provided')
    
    else:
        arg = np.argmin(day_deltas)

        if arg.size != 1: arg = arg[0]

        datestr = files[arg][22:22+2+2+4]

        return f'{year}:{datestr[:2]}:{datestr[2:4]}:{datestr[4:]}'

def model(date:str,
          delim:str=':',
          days_around:int|float=14,
          modeltype:str='PFSS',
          **modelkwargs):
    utils.type_check(locals(),model)

    datetime = utils.parse_to_datetime(date,delim=delim)

    if modeltype == 'PFSS':
        modeldate = find_close_magneto_date(date,delim=delim,days_around=days_around)
        oflname, nlname, rss = pfss(modeldate,delim=delim,return_name=True,**modelkwargs)
        modelkwargs['rss'] = rss

    else: raise NameError(f"Model '{modeltype}' is not implemented")

    datetime_model = utils.parse_to_datetime(modeldate)

    chmodel_path = f'{__path__[0]}/out/{oflname}'
    nlmodel_path = f'{__path__[0]}/out/{nlname}'

    return CoronalModel(modeltype=modeltype,
                        datetime=datetime,
                        datetime_model=datetime_model,
                        chmodel_path=chmodel_path,
                        nlmodel_path=nlmodel_path,
                        modelkwargs=modelkwargs)

class CoronalModel:
    def __init__(self,
                 modeltype:str,
                 datetime:datetime.datetime,
                 datetime_model:datetime.datetime,
                 chmodel_path:str,
                 nlmodel_path:str,
                 modelkwargs):
        utils.type_check(locals(),CoronalModel.__init__)
        
        self.modeltype = modeltype
        
        if 'rss' in modelkwargs: self.rss = modelkwargs['rss']

        self.datetime = datetime
        self.datetime_model = datetime_model

        self.chmodel_path = chmodel_path
        self.nlmodel_path = nlmodel_path


    def chmodel(self): return utils.csv2map(self.chmodel_path, self.datetime_model)

    def nlmodel(self): return utils.csv2map(self.nlmodel_path, self.datetime_model)

    def chmetric(self, replace:bool=False, days_around:int=14):
        euvmappath = chmetric.create_euv_map(self.datetime, 
                                                days_around=days_around, replace=replace)
        euvmap = sunpy.map.Map(euvmappath)
        
        thresh1, thresh2 = thresholds(euvmap)

        ch_obs_path = chmetric.extract_obs_ch(  
                                                euvmappath,
                                                replace=replace,
                                                ezseg_version='fortran',
                                                ezseg_params={
                                                    "thresh1":thresh1,#np.nanmax(euvmap.data.flatten())*0.07, ## Seed threshold
                                                    "thresh2":thresh2,#np.nanmax(euvmap.data.flatten())*0.101, ## Growing Threshhold
                                                    "nc":5, ## at least 5 consecutive pixels to declare coronal hole area is connected
                                                    "iters":100   
                                                }
                                            )

        p,r,f = chmetric.do_ch_score(self.datetime_model,
                                self.chmodel_path,
                                ch_obs_path,
                                auto_interp=True)

        return p,r,f
    
    def plot_chmetric(self, replace:bool=False, days_around:int=14):
        datetime_euvmap = self.datetime
        euvmappath = chmetric.create_euv_map(datetime_euvmap, 
                                                days_around=days_around, replace=replace)
        euvmap = sunpy.map.Map(euvmappath)
        
        thresh1, thresh2 = thresholds(euvmap)

        ch_for_path = chmetric.extract_obs_ch(euvmappath,
                                        replace=replace,
                                        ezseg_version='fortran',
                                        ezseg_params={
                                            "thresh1":thresh1,#np.nanmax(euvmap.data.flatten())*0.07, ## Seed threshold
                                            "thresh2":thresh2,#np.nanmax(euvmap.data.flatten())*0.101, ## Growing Threshhold
                                            "nc":5, ## at least 5 consecutive pixels to declare coronal hole area is connected
                                            "iters":100   
                                        }
                                        )
        
        ch_for_map = sunpy.map.Map(ch_for_path)
        lognorm = mpl.colors.LogNorm(vmin=np.nanpercentile(euvmap.data.flatten(),5), 
                                vmax=np.nanpercentile(euvmap.data.flatten(),99.9))

        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot(projection=euvmap.wcs)
        euvmap.plot(cmap="sdoaia193",
                    norm=lognorm,
                    axes=ax
                )
        ch_for_map.draw_contours(levels=[.5,], colors=["white"], axes=ax)

        ax.set_title("Fortran/CHMAP EZSEG CHD: " + euvmap.meta['date-obs'][:-13]);

        ### We have some sample coronal hole and neutral line maps in the folder:
        # ./example_model_data/
        # Let's choose one
        ch_obs_map = ch_for_map # fortran map

        ### The model result y-axis is binned in the Cylindrical equal area projection
        # while the observation is binned in latitude. We can convert one to the other
        # by using the sunpy reprojection api (this also will interpolate the map to 
        # the same resolution as the model result, which we also need for doing the
        # pixel by pixel classification )

        ch_obs_cea = ch_obs_map.reproject_to(self.chmap_model.wcs)
        ch_combined = sunpy.map.Map(
            ch_obs_cea.data+self.chmap_model.data,
            self.chmap_model.meta
        )
        ## Now we can plot this side by side with "observed" coronal holes and see
        ## how they compare
        fig = plt.figure()
        plt.imshow(ch_combined.data,cmap='inferno')
        plt.title(f'chmetric {self.date}')
        plt.xlabel('Carrington Longitude')
        plt.ylabel('Latitude')
    
    def wlmetric(self, quiet:bool=True, method:str='Simple'):
        from importlib import reload;reload(cmfpy.io)
        #### Capability to create locally pending

        #### Load Precomputed Ones

        ### Either download from online source (2020.4.27-2023) or download local 
        # Load location is determined by input date.

        WL_date = self.datetime
        WL_path = f"{wlmetric.__path__[0]}/data"
        if self.datetime < datetime.datetime(2020,4,27): WL_source = 'V3'
        else: WL_source = 'connect_tool'

        [WL_fullpath,WL_date] = cmfpy.io.get_WL_map(WL_date,
                                                WL_path,
                                                WL_source,
                                                quiet=quiet)

        wlmap = cmfpy.io.WLfile2map(WL_fullpath,WL_date,WL_source)


        score = wlmetric.compute_WL_score(self.nlmodel(),wlmap,method=method)
        return score/100
    
    def plot_wlmetric(self, quiet:bool=True,method='Simple'):
        from importlib import reload;reload(cmfpy.io)
        #### Capability to create locally pending

        #### Load Precomputed Ones

        ### Either download from online source (2020.4.27-2023) or download local 
        # Load location is determined by input date.

        WL_date = self.datetime

        WL_path = f"{wlmetric.__path__[0]}/data"
        if self.datetime < datetime.datetime(2020,4,27): WL_source = 'V3'
        else: WL_source = 'connect_tool'
        
        [WL_fullpath,WL_date] = cmfpy.io.get_WL_map(WL_date,
                                                WL_path,
                                                WL_source,
                                                quiet=quiet)

        wlmap = cmfpy.io.WLfile2map(WL_fullpath,WL_date,WL_source)
        nlmodel = self.nlmodel()
        
        ### INPUT : sunpy.map.Map from a White Light Carrington Map
        ### OUTPUT : astropy.coordinates.SkyCoord describing the 
        # location of maximum brightness in the image at each longitude
        smb, _ = wlmetric.extract_SMB(wlmap,method=method)


        ### As with the chmetric, the observed map is in constant latitude 
        # binning. For a fair comparison, we reproject to sine(latitude)
        # (CEA) binning.
        wl_obs_cea = wlmap.reproject_to(nlmodel.wcs)

        ## Now we can plot the model and observations side by side
        fig = plt.figure(figsize=(20,5))
        axmodel = fig.add_subplot(131,projection=nlmodel.wcs) 
        axobs = fig.add_subplot(132,projection=wl_obs_cea.wcs)
        axcomp = fig.add_subplot(133,projection=nlmodel.wcs)

        nlmodel.plot(cmap="coolwarm",axes=axmodel)
        nlmodel.draw_contours(levels=[0],colors=["black"],axes=axmodel)
        wl_obs_cea.plot(cmap="Greys_r",axes=axobs)
        axobs.plot_coord(smb,"o",color="gold",ms=1)
        nlmodel.plot(cmap="coolwarm",axes=axcomp,vmin=-1.5,vmax=1.5)
        nlmodel.draw_contours(levels=[0],colors=["black"],axes=axcomp,label="Model")
        axcomp.plot_coord(smb,"o",color="gold",ms=1,label="Observed")

        axmodel.set_title("Modeled Neutral Line")
        axobs.set_title("Observed Streamer Belt")
        axcomp.set_title("Comparison")

    def nlmetric(self):
        observed_field_l1 = nlmetric.create_polarity_obs(self.datetime,"L1",return_br=True)
        polarity_pred_l1 = nlmetric.create_polarity_model(self.nlmodel(),self.datetime_model,"L1",altitude=self.rss*u.R_sun)

        ######################
        
        ### Finally we can go ahead and combine the predicted and measured timeseries
        # to produce our nlmetric score for each spacecraft we compare. 

        l1_nl_score = nlmetric.compute_NL_metric(polarity_pred_l1,observed_field_l1)

        return l1_nl_score
    
    def plot_nlmetric(self):
        nlmodel = self.nlmodel()
        observed_field_l1 = nlmetric.create_polarity_obs(self.datetime,"L1",return_br=True)
        polarity_pred_l1 = nlmetric.create_polarity_model(nlmodel,self.datetime_model,"L1")
        
        ######################

        ## Now we have produced predicted and measured polarity timeseries, so we can visualize the comparison.

        ## First, directly by looking at the timeseries
        fig,axes = plt.subplots(figsize=(12,4),ncols=2)

        axes[0].plot(*polarity_pred_l1,color="black",linewidth=2,label="Predicted")
        axes[0].plot(observed_field_l1[0],np.sign(observed_field_l1[1]),color="red",label="Observed")
        axes[0].set_title("L1/Wind");axes[1].set_ylabel("Magnetic Polarity")

        for ax in axes: 
            ax.legend()
            ax.set_ylabel("Magnetic Polarity")
            ax.set_yticks([-1,1])

        ## Second, plot in an easier way to view/understand using the map and the trajectories

        # We first need to use some of the helpers to create the trajectories
        
        
        carrington_trajectory_l1 = projection.create_carrington_trajectory(
            polarity_pred_l1[0],"L1",obstime_ref=self.datetime_model
            )
        trajectory_l1 = projection.ballistically_project(carrington_trajectory_l1,
                                                    r_inner=self.rss*u.R_sun) 


        fig=plt.figure(figsize=(12,6))
        ax = fig.add_subplot(projection=nlmodel.wcs)
        nlmodel.plot(cmap="coolwarm",vmin=-2,vmax=2)


        ax.scatter(nlmodel.world_to_pixel(trajectory_l1).x,
                nlmodel.world_to_pixel(trajectory_l1).y,
                c=plt.cm.bwr(observed_field_l1[1]),s=2
                )