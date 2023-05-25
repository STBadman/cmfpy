'''
This module shall provide capability to :
1) <CREATE MEASURED BR TIMESERIES> Given input spacecraft and timestamp, compute 1 hour 
most probable polarity for 1 carrington rotation centered on timestamp
2) <CREATE PREDICTED BR TIMESERIES> Given model NL, input spacecraft and timestamp, compute predicted timeseries of polarity  
3) <DO NL SCORE> Read in measured and predicted Br, compute NL Score
'''
import astropy.units as u
import datetime
import copy
import numpy as np
import helpers as h
import os
### Change pyspedas directory to NLmetric/data
os.environ['SPEDAS_DATA_DIR']="./NLmetric/data/"
import pyspedas
import sunpy.coordinates
import sys

def determine_carrington_interval(center_date,body) :
    inst_body_position = h.create_carrington_trajectory(
        [center_date],body
        )
    inst_lon = inst_body_position.lon
    inst_antipode = (inst_body_position.lon.value + 180 % 360) *u.deg
    two_month_window = h.gen_dt_arr(
        center_date-datetime.timedelta(days=30),
        center_date+datetime.timedelta(days=30),
        cadence_days=6/24
        )
    two_month_trajectory = h.create_carrington_trajectory(
        two_month_window,body
        )
    carr_inds = np.where(np.diff(
        (two_month_trajectory.lon-inst_antipode).value % 360
        ) > 180)[0] + 1

    carrington_interval = two_month_window[carr_inds]

    return carrington_interval

def download_br_data(interval, body) :
    dl_funcs_and_kwargs = {
        "L1":(
            pyspedas.wind.mfi,
            {"varnames":"BGSE"}
            ),
        "solar orbiter":(
            pyspedas.solo.mag,
            {"datatype":"rtn-normal-1-minute"}
            ),
        "psp":(
            pyspedas.psp.fields,
            {"datatype":"mag_rtn_1min"}
            ),
        "stereo-a":(
            pyspedas.stereo.mag,
            {"probe":"a"}

        ),
        "stereo-b":(
            pyspedas.stereo.mag,
            {"probe":"b"}
        )
    }
    dl_func,kwargs = dl_funcs_and_kwargs.get(body)
    return dl_func(trange=interval,**kwargs,notplot=True)

def make_hourly_medians(datetimes,data) :
    timestamps = np.array([t.timestamp() for t in datetimes])  
    datetime_hourly = h.gen_dt_arr(
        datetimes[0],
        datetimes[-1],
        cadence_days=1/24
        )  
    ts_edges = np.array([t.timestamp() for t in datetime_hourly[1:-1]])
    
    argsplit = [np.where(timestamps > te)[0][0] for te in ts_edges]
    data_split = np.split(data,argsplit)
    medians = np.array([np.nanmedian(slice_) for slice_ in data_split])
    return  (datetime_hourly[:-1]+datetime.timedelta(hours=0.5),
             medians)


def create_br_obs(center_date,body,save_dir="./NLmetric/data/") :
    '''
    Given `center_date`:`datetime.datetime` and `spacecraft`*:`str`,
    1) determine the time interval required to span a Carrington 
    rotation worth of data, producing t_start, t_end
    2) download measured magnetic field data from the relevant
     spacecraft* between those dates
    3) For every hour, compute a histogram of measurements and find the
    most probable measurement, take its sign to obtain polarity
    4) Return timeseries and save in save_dir.

    * should validate spacecraft spice kernels are accessible via
    astrospice
    '''
    carrington_interval = determine_carrington_interval(
        center_date,body
        )
    data = download_br_data(carrington_interval, body)
    
    times_hourly,br_hourly = make_hourly_medians(
        data[list(data.keys())[0]]['x'],
        data[list(data.keys())[0]]['y'][:,0],
    )
    if body == "L1" : br_hourly *= -1
    return times_hourly,br_hourly

def create_br_model(model_NL_map, center_date, spacecraft,
                    altitude=2.5*u.R_sun,save_dir="./") :
    '''
    Given `model_NLmap` (modeled neutral line map user provided), 
    `center_date`:`datetime.datetime` (which should match the 
    magnetogram date of the neutral line map) and an astrospice-valid
    `spacecraft`:`str`: 
    1) Determine the time interval required to span a Carrington
    rotation worth of data, producing `t_start`, `t_end`
    2) construct a ballistically mapped trajectory of the spacecraft 
    at the altitude of the model NL map (default 2.5Rs) between those
    two dates. 
    3) Fly spacecraft over the NL map and sample the predicted magnetic
    polarity as a function of time
    4) Return predicted timeseries and save in save_dir
    '''
    pass

def compute_NL_metric(model_tseries,obs_tseries) :
    '''
    Given `model_tseries` and `obs_tseries`, ensure the timestamps are
    aligned, compute the dot product of the data, and divide by the
    number of the datapoints to obtain the NL_metric score.
    '''
    pass

'''
import numpy as np

def lookupNearest(x0, y0, x, y, data):
     xi = np.abs(x-x0).argmin()
     yi = np.abs(y-y0).argmin()
     return data[yi,xi]
'''


'''
dt_E1 = pp.gen_dt_arr(datetime(2018,10,1),datetime(2018,11,30))
dt_E2 = pp.gen_dt_arr(datetime(2019,3,1),datetime(2019,4,30))
dt_E3 = pp.gen_dt_arr(datetime(2019,8,1),datetime(2019,9,30))
dtlims_dict = {1:[datetime(2018,10,1),datetime(2018,11,30)],
               2:[datetime(2019,3,1),datetime(2019,4,30)],
               3:[datetime(2019,8,1),datetime(2019,9,30)]}



#filename = "/home/sam/1_RESEARCH/FIELDS/ISSI_PAPER/in_situ_comparison/in_situ.pkl"
filename = "/home/sam/1_RESEARCH/FIELDS/ISSI_PAPER/metric_data/Pol/pfss_pol_metric_dat.pkl"
if not os.path.exists(filename) :    
    out_dict = {}
    for peri_num, dt_arr in zip([1,2,3],[dt_E1,dt_E2,dt_E3]) :
        sys.stdout.write(f"Encounter {peri_num}\n")
        out_dict_ = {}
        for rss in [1.5,2.0,2.5,3.0] :
            sys.stdout.write(f"Rss = {rss}\n")
            bpsp_pred_agong,bsta_pred_agong,bwind_pred_agong = [],[],[]
            bpsp_pred_ahmi,bsta_pred_ahmi,bwind_pred_ahmi = [],[],[]
            bpsp_pred_mrzqs,bsta_pred_mrzqs,bwind_pred_mrzqs = [],[],[]

            psp_ss = pp.get_ss_footpoints(dt_arr)
            wind_ss = pp.get_ss_footpoints(dt_arr,spice_str="L1")
            sta_ss = pp.get_ss_footpoints(dt_arr, spice_str="STEREO AHEAD")

            dtlims = dtlims_dict.get(peri_num)
            (t_common,Br_psp,Vr_psp,psp_coords,
                  Br_sta,Vr_sta,sta_coords,
                  Br_wind,Vr_wind,wind_coords) = pp.load_all_interp(peri_num,interp_cadence=1.0,
                                                                    t_start = dtlims[0],                                                                   t_end = dtlims[1]
                                                                   )
            
            for dt,psp_,wind_,sta_ in zip(dt_arr,psp_ss,wind_ss,sta_ss) :
                sys.stdout.write(f"{str(dt)[0:15]}...\r")
                (lonagong,slatagong,agong_NL,
                lonahmi,slatahmi,ahmi_NL,
                lonmrzqs,slatmrzqs,mrzqs_NL) = load_NL(datetime(dt.year,
                                                                dt.month,
                                                                dt.day), 
                                                       rss, peri_num)   
                bpsp_pred_agong.append(lookupNearest(
                    psp_.lon.value,np.sin(psp_.lat),
                    lonagong,slatagong,agong_NL.values
                ))
                bsta_pred_agong.append(lookupNearest(
                    sta_.lon.value,np.sin(sta_.lat),
                    lonagong,slatagong,agong_NL.values
                ))
                bwind_pred_agong.append(lookupNearest(
                    wind_.lon.value,np.sin(wind_.lat),
                    lonagong,slatagong,agong_NL.values
                ))

                bpsp_pred_ahmi.append(lookupNearest(
                    psp_.lon.value,np.sin(psp_.lat),
                    lonahmi,slatahmi,ahmi_NL.values
                ))
                bsta_pred_ahmi.append(lookupNearest(
                    sta_.lon.value,np.sin(sta_.lat),
                    lonahmi,slatahmi,ahmi_NL.values
                ))
                bwind_pred_ahmi.append(lookupNearest(
                    wind_.lon.value,np.sin(wind_.lat),
                    lonahmi,slatahmi,ahmi_NL.values
                ))

                bpsp_pred_mrzqs.append(lookupNearest(
                    psp_.lon.value,np.sin(psp_.lat),
                    lonmrzqs,slatmrzqs,mrzqs_NL
                ))
                bsta_pred_mrzqs.append(lookupNearest(
                    sta_.lon.value,np.sin(sta_.lat),
                    lonmrzqs,slatmrzqs,mrzqs_NL
                ))
                bwind_pred_mrzqs.append(lookupNearest(
                    wind_.lon.value,np.sin(wind_.lat),
                    lonmrzqs,slatmrzqs,mrzqs_NL
                ))



            sys.stdout.write("\n")
            out_dict_[rss] = {"PSP":{"PFSS-AGONG" : bpsp_pred_agong,
                                     "PFSS-AHMI" : bpsp_pred_ahmi,
                                     "PFSS-GONGz" : bpsp_pred_mrzqs 
                                    },
                              "STA":{"PFSS-AGONG" : bsta_pred_agong ,
                                     "PFSS-AHMI" : bsta_pred_ahmi,
                                     "PFSS-GONGz" : bsta_pred_mrzqs 
                                    },
                              "Wind":{"PFSS-AGONG" : bwind_pred_agong ,
                                     "PFSS-AHMI" : bwind_pred_ahmi ,
                                     "PFSS-GONGz" : bwind_pred_mrzqs 
                                    }
                             } 
        out_dict[peri_num] = [out_dict_,{"PSP":Br_psp,"STA":Br_sta,"Wind":Br_wind}]
    pickle.dump(out_dict,open(filename,"wb"))
else : out_dict = pickle.load(open(filename,"rb"))
'''