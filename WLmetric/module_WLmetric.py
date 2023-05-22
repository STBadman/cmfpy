import numpy as np
from scipy.ndimage import gaussian_filter1d

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

def compute_WLscore(x,mean_val,err_val):
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

#Score functions definitions
def sigmoid(x,a,b):
    y = 1/(1+np.exp(-b*(x+a)));
    return(y)