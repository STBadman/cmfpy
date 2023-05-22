'''
This module shall provide capability to :
1) <CREATE MEASURED BR TIMESERIES> Given input spacecraft and timestamp, compute 1 hour 
most probable polarity for 1 carrington rotation centered on timestamp
2) <CREATE PREDICTED BR TIMESERIES> Given model NL, input spacecraft and timestamp, compute predicted timeseries of polarity  
3) <DO NL SCORE> Read in measured and predicted Br, compute NL Score
'''

import numpy as np

def lookupNearest(x0, y0, x, y, data):
     xi = np.abs(x-x0).argmin()
     yi = np.abs(y-y0).argmin()
     return data[yi,xi]



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