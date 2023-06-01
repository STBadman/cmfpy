'''
This module shall provide capability to :
1) <CREATE MAP> Create a WL Carrington map assimilating SOHO/LASCO data over one carrington rotation, save as fits file
2) <EXTRACT OBSERVED NL> Load a pre-computed fits file and apply Poirier routines to extract contour 
3) <DO WL SCORE> Read in an "observed NL" map and "modeled NL" map, compute normalized distance between curves
'''

import astropy.units as u
from astropy.coordinates import SkyCoord
import numpy as np
import os
import WLmetric.io_functions # functions to read in maps from different sources
from scipy.ndimage import gaussian_filter1d
import sunpy.map
from scipy.interpolate import interp1d

#Score functions definitions
def sigmoid(x,a,b):
    y = 1/(1+np.exp(-b*(x+a)));
    return(y)

def create_WL_map(center_date, coronagraph_altitude=3*u.R_sun) :
    '''
    *** TO BE IMPLEMENTED ***
    Given `center_date`:`datetime.datetime`, download SOHO/LASCO
    coronagraph images for one Carrington rotation centered on that
    date, extract white light intensity above the limbs at the 
    specified altitude, and assemble into a Carrington map. Save as
    fits file.
    '''
    pass

def extract_edge_coords(wl_map) :

    WL_I = wl_map.data

    #Reconstruct the grid coordinates
    WL_phi_edges = np.linspace(0,360,num=np.size(WL_I,1)+1) #pixel edges
    WL_th_edges = np.linspace(-90,90,num=np.size(WL_I,0)+1) #pixel edges
    [WL_pphi_edges, WL_tth_edges] = np.meshgrid(WL_phi_edges, WL_th_edges)
    WL_phi = (WL_phi_edges[1:]+WL_phi_edges[0:-1])/2 #pixel centres
    WL_th = (WL_th_edges[1:]+WL_th_edges[0:-1])/2 #pixel centres
    [WL_pphi, WL_tth] = np.meshgrid(WL_phi, WL_th)
    
    #Input image is cell-centered -> conversion to edge values
    WL_I_edges = np.empty((np.size(WL_th_edges,0),np.size(WL_phi_edges,0)))
    WL_I_edges[1:-1,1:-1] = (WL_I[:-1,:-1]+WL_I[1:,1:])/2
    WL_I_edges[1:-1,0] = (WL_I[:-1,0]+WL_I[1:,-1])/2
    WL_I_edges[1:-1,-1] = (WL_I[:-1,0]+WL_I[1:,-1])/2
    WL_I_edges[0,1:-1] = (WL_I[0,:-1]+WL_I[-1,1:])/2
    WL_I_edges[-1,1:-1] = (WL_I[0,:-1]+WL_I[-1,1:])/2
    WL_I_edges[[0,0,-1,-1],[0,-1,0,-1]] = 0.

    return WL_pphi_edges, WL_tth_edges, WL_I_edges


def extract_SMB(wl_map,smoothing_factor=20,
                save_dir=os.path.join("WLmetric","data")
                ):
    '''
    Given a precomputed input White light carrington map (`wl_map`),
    extract the streamer maximum brightness (SMB) line, and the 
    half brightness contour as a function of longitude. (Following 
    Poirier+2021, Badman+2022). Save in `save_dir` 
    '''
    WL_pphi_edges,WL_tth_edges,WL_I_edges = extract_edge_coords(wl_map)
    
    #Find maximal brightness for each longitude
    idx_max = np.nanargmax(WL_I_edges,axis=0)

    #Fetch coordinates of these maxima
    SMB_phi_edges = WL_pphi_edges[0,:].flatten()
    SMB_th_edges = WL_tth_edges[idx_max,0].flatten()

    #Before smoothing the curve, repeat data at each side
    n_extend = int(smoothing_factor/2)+1;
    SMB_th_extended = np.concatenate((SMB_th_edges[-1-(n_extend-1):-1],
                                      SMB_th_edges,
                                      SMB_th_edges[1:n_extend-1]),axis=0);

    #Smooth the SMB line
    SMB_th_extended = gaussian_filter1d(SMB_th_extended,smoothing_factor)

    #Shrink back to initial range
    SMB_th_edges = SMB_th_extended[(n_extend-1):-(n_extend-1)+1];
    return SkyCoord(lon=SMB_phi_edges*u.deg,
                    lat=SMB_th_edges*u.deg,
                    radius=[1*u.R_sun]*len(SMB_phi_edges),
                    frame="heliographic_carrington",
                    observer=wl_map.observer_coordinate,
                    obstime=wl_map.observer_coordinate.obstime,
                    representation_type="spherical"
                    )

#def compute_min_dist(model_map, wl_map) :
def compute_min_dist(phi1,th1,phi2,th2):

    #Cpmpute angular distance (i.e. central angle) between all these points
    #see: https://en.wikipedia.org/wiki/Great-circle_distance
    phi1=phi1.to("rad").value
    th1=th1.to("rad").value
    phi2=phi2.to("rad").value
    th2=th2.to("rad").value

    dphi = np.abs(phi2-phi1)
    dth = np.abs(th2-th1)
    sigma = np.arctan2(np.sqrt((np.cos(th2)*np.sin(dphi))**2+(np.cos(th1)*np.sin(th2)-np.sin(th1)*np.cos(th2)*np.cos(dphi))**2),
                       (np.sin(th1)*np.sin(th2)+np.cos(th1)*np.cos(th2)*np.cos(dphi))
                       )*u.rad

    #Find the minimum angle distances between NL and SMB
    sigma_min = np.min(sigma,axis=0)
    idx_min = np.argmin(sigma,axis=0) #just for plotting

    return(sigma_min.to("deg"),idx_min)

def compute_WL_score(model_nl_map,obs_wl_map,norm_mode="fixed") :
    '''
    Given `model_nl_map` (user provided) and a precomputed `smb_obs`
    dataset describing the coronagraph-observed neutral line, extract
    the 1d model neutral line, and apply the Poirier+ method to compute
    the average angular distance between the two 1d curves weighted by
    the local streamer belt thickness. 
    '''
    norm_mode_all = ["fixed","SB_thickness"]
 
    ### Extract two curves to compare
    nl_model = model_nl_map.contour([0])[0] ### ASSUMES ONLY ONE CONTOUR, CAUTION!
    nl_lon,nl_lat = nl_model.lon.value,nl_model.lat.value
    smb_obs = extract_SMB(obs_wl_map)
    smb_lon,smb_lat = smb_obs.lon.value,smb_obs.lat.value

    ### Find the closest pair of points on each curve and compute the angular distance
    phi2 = np.tile(nl_lon[:,np.newaxis],(1,np.size(smb_lon,0)))*u.deg
    th2 = np.tile(nl_lat[:,np.newaxis],(1,np.size(smb_lon,0)))*u.deg
    phi1 = np.tile(smb_lon[:,np.newaxis].T,(np.size(nl_lon,0),1))*u.deg
    th1 = np.tile(smb_lat[:,np.newaxis].T,(np.size(nl_lon,0),1))*u.deg
    [min_separation,idx_min] = compute_min_dist(phi1,th1,phi2,th2) 
    #return sigma_min

    if norm_mode=='fixed': #i.e. equivalent to a constant streamer belt thickness
            norm_val = 5*np.ones(np.shape(min_separation)) #in deg
    elif norm_mode=='SB_thickness': #Compute real streamer belt thickness from WL map
            print("Computation of the real streamer belt thickness is not implemented yet!! Switching to fixed mode: thick=5deg.")
            norm_val = 5*np.ones(np.shape(min_separation)) #in deg
    else : raise ValueError(f"norm_mode {norm_mode} not in {norm_mode_all}")
    min_separation/=norm_val

    return eval_WL_score(min_separation)


def eval_WL_score(min_separation_normed) :
    """
    Do score evaluation based on set of minimum distances between curves, 
    as defined in Poirier et al. (2021)
    """

    mean_val = np.mean(min_separation_normed).to("deg").value
    err_val = np.std(min_separation_normed).to("deg").value

    #---- Definition of the gain function ----
    #Define constant c1 (horizontal shift)
    #c1 = -1 #as published in [Poirier et al. 2021] => gain=0.5 for mean_val=1
    c1 = -2;
    
    #Define constant c2 (slope of the sigmoid)
    #c2 = -np.log(1/0.01-1)/(2+c1); #as published in [Poirier et al. 2021] => gain=0.01 for mean_val=2
    c2 = -np.log(1/0.01-1)/(4+c1);
    gain = sigmoid(mean_val,c1,c2);
    
    #---- Definition of the penalty function ----
    #Define constant c3 (horizontal shift)
    #c3 = -1 #as published in [Poirier et al. 2021] => penalty=0.5 for err_val=1
    c3 = -2;
    
    #Define constant c4 (slope of the sigmoid)
    #c4 = -np.log(1/0.99-1)/(2+c3); #as published in [Poirier et al. 2021] => penalty=0.99 for err_val=2
    c4 = -np.log(1/0.99-1)/(4+c3);
    penalty = sigmoid(err_val,c3,c4);
    
    #---- Compute final score ----
    score = gain*(1 - penalty);
    score = 100*score; #output in % (100% is a perfect model, 0% is the worst)

    return score
