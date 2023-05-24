'''
This module shall provide capability to :
1) <CREATE MAP> Create a WL Carrington map assimilating SOHO/LASCO data over one carrington rotation, save as fits file
2) <EXTRACT OBSERVED NL> Load a pre-computed fits file and apply Poirier routines to extract contour 
3) <DO WL SCORE> Read in an "observed NL" map and "modeled NL" map, compute normalized distance between curves
'''

import astropy.units as u
import WLmetric.io_functions # functions to read in maps from different sources

#Score functions definitions
def sigmoid(x,a,b):
    y = 1/(1+np.exp(-b*(x+a)));
    return(y)

def create_WL_map(center_date, coronagraph_altitude=3*u.R_sun) :
    '''
    Given `center_date`:`datetime.datetime`, download SOHO/LASCO
    coronagraph images for one Carrington rotation centered on that
    date, extract white light intensity above the limbs at the 
    specified altitude, and assemble into a Carrington map. Save as
    fits file.
    '''
    pass

def extract_SMB(wl_map,save_dir='./') :
    '''
    Given a precomputed input White light carrington map (`wl_map`),
    extract the streamer maximum brightness (SMB) line, and the 
    half brightness contour as a function of longitude. (Following 
    Poirier+2021, Badman+2022). Save in `save_dir` 
    '''
    pass

def compute_SMB(WL_I_edges,WL_pphi_edges,WL_th_edges,smoothing_factor):
    #Find maximal brightness for each longitude
    idx_max = np.nanargmax(WL_I_edges,axis=0)

    #Fetch coordinates of these maxima
    SMB_phi_edges = WL_pphi_edges[0,:].flatten()
    SMB_th_edges = WL_th_edges[idx_max,0].flatten()

    #Before smoothing the curve, repeat data at each side
    n_extend = int(smoothing_factor/2)+1;
    SMB_th_extended = np.concatenate((SMB_th_edges[-1-(n_extend-1):-1],SMB_th_edges,SMB_th_edges[1:n_extend-1]),axis=0);

    #Smooth the SMB line
    SMB_th_extended = gaussian_filter1d(SMB_th_extended,smoothing_factor)

    #Shrink back to initial range
    SMB_th_edges = SMB_th_extended[(n_extend-1):-(n_extend-1)+1];

    return(SMB_phi_edges,SMB_th_edges)

def compute_min_dist(phi1,th1,phi2,th2):
    #Cpmpute angular distance (i.e. central angle) between all these points
    #see: https://en.wikipedia.org/wiki/Great-circle_distance
    dphi = np.abs(phi2-phi1)
    dth = np.abs(th2-th1)
    sigma = np.arctan2(np.sqrt((np.cos(th2)*np.sin(dphi))**2+(np.cos(th1)*np.sin(th2)-np.sin(th1)*np.cos(th2)*np.cos(dphi))**2),(np.sin(th1)*np.sin(th2)+np.cos(th1)*np.cos(th2)*np.cos(dphi)))
    sigma *= 180/np.pi

    #Find the minimum angle distances between NL and SMB
    sigma_min = np.min(sigma,axis=0)
    idx_min = np.argmin(sigma,axis=0) #just for plotting

    return(sigma_min,idx_min)

def compute_WL_score(model_nl_map,smb_obs) :
    '''
    Given `model_nl_map` (user provided) and a precomputed `smb_obs`
    dataset describing the coronagraph-observed neutral line, extract
    the 1d model neutral line, and apply the Poirier+ method to compute
    the average angular distance between the two 1d curves weighted by
    the local streamer belt thickness. 
    '''

    #


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

    return(score)


    pass