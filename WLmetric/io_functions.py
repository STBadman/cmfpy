import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from astropy.io import fits
import os
from datetime import datetime
import wget
import zipfile
import requests
import sunpy.map
from sunpy.coordinates import get_earth

def read_NL_SAM(NL_fullpath): 
    with open(NL_fullpath, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        BSS_th = np.array([])
        c=0
        for row in spamreader:
            if c==0:
                row = row[1:]
                BSS_phi = np.array([float(i) for i in row])
            else:
                row = np.array([float(i) for i in row])
                BSS_th = np.concatenate((BSS_th,[row[0]]),axis=0)
                if c==1:
                    BSS_pol = row[1:]
                    BSS_pol = BSS_pol[:,np.newaxis].T
                else:
                    BSS_pol = np.concatenate((BSS_pol,row[1:,np.newaxis].T),axis=0)
            c+=1
        BSS_th = np.arcsin(BSS_th)*180/np.pi
        [BSS_pphi, BSS_tth] = np.meshgrid(BSS_phi, BSS_th)
    
    #Extract NL from BSS map
    fig_tmp, ax_tmp = plt.subplots()
    cs = ax_tmp.contour(BSS_pphi,BSS_tth,BSS_pol, [0.])
    p = cs.collections[0].get_paths()[0]
    v = p.vertices
    NL_phi_edges = v[:,0]
    NL_th_edges = v[:,1]
    plt.close(fig_tmp)
    
    return(NL_phi_edges,NL_th_edges)

def read_WL_map(WL_fullpath,source):
    sources = ['connect_tool',"V1.1"]
    if source=='connect_tool':
        [WL_I,
         WL_pphi,
         WL_tth,
         WL_I_edges,
         WL_pphi_edges,
         WL_tth_edges]=read_WL_map_connecttool(WL_fullpath)
    elif source=='V1.1':
        [WL_I,
         WL_pphi,
         WL_tth,
         WL_I_edges,
         WL_pphi_edges,
         WL_tth_edges]=read_WL_map_V1p1(WL_fullpath)
    else : raise ValueError(f"Source {source} not in {sources}")
    return(WL_I,WL_pphi,WL_tth,WL_I_edges,WL_pphi_edges,WL_tth_edges)

def WLfile2map(WL_fullpath,WL_date,WL_source):
    if WL_source == "connect_tool" :
        WL_I = mpimg.imread(WL_fullpath) #cell-centered valued

        # Convert from RGB to grayscale
        def rgb2gray(I):
            return np.dot(I[...,:3], [0.2989, 0.5870, 0.1140])
        WL_I = rgb2gray(WL_I)[::-1,:]

    elif WL_source == "V1.1" :
        WL_I_edges = fits.getdata(WL_fullpath,ext=0)
        
        #If latitude columns are full of nan, replace by zeros
        idx_nan = ~np.any(~np.isnan(WL_I_edges),axis=0)
        WL_I_edges[np.tile(idx_nan[:,np.newaxis].T,
                           (WL_I_edges.shape[0],1))] = 0.

        WL_I = (WL_I_edges[:-1,:-1]+WL_I_edges[1:,1:])/2

    header = sunpy.map.header_helper.make_heliographic_header(
        WL_date, get_earth(WL_date),WL_I.shape, frame='carrington'
    )
    ## Fix header metadata to place CR0 at LH edge of map
    header['crval1']=180.0
    
    return sunpy.map.Map(WL_I,header)
        
def read_WL_map_connecttool(WL_fullpath):
    WL_I = mpimg.imread(WL_fullpath) #cell-centered valued

    #Convert from RGB to grayscale
    def rgb2gray(I):
        return np.dot(I[...,:3], [0.2989, 0.5870, 0.1140])
    WL_I = rgb2gray(WL_I)    
    
    #Reconstruct the grid coordinates
    WL_phi_edges = np.linspace(0,360,num=np.size(WL_I,1)+1) #pixel edges
    WL_th_edges = np.linspace(90,-90,num=np.size(WL_I,0)+1) #pixel edges
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
    
    return(WL_I,WL_pphi,WL_tth,WL_I_edges,WL_pphi_edges,WL_tth_edges)

def read_WL_map_V1p1(WL_fullpath):
    WL_I_edges = fits.getdata(WL_fullpath,ext=0) #edge-valued
    header = fits.getheader(WL_fullpath,ext=0)
    
    #Reconstruct the grid coordinates
    NAXIS1 = int(header['NAXIS1']) #Nb points along longitude axis
    NAXIS2 = int(header['NAXIS2']) #Nb points along latitude axis
    CRPIX1 = float(header['CRPIX1']) #Idx of the reference pixel on longitude
    CRPIX2 = float(header['CRPIX2']) #Idx of the reference pixel on latitude
    CRVAL1 = float(header['CRVAL1']) #Carrington longitude of reference pixel in arcsec
    CRVAL2 = float(header['CRVAL2']) #Latitude of reference pixel in arcsec
    CDELT1 = float(header['CDELT1']) #Longitude angle resolution in arcsec
    CDELT2 = float(header['CDELT2']) #Latitude angle resolution in arcsec
    WL_phi_edges = np.array([(i+1-CRPIX1)*CDELT1+CRVAL1 for i in range(NAXIS1)])/3600 #in deg
    WL_th_edges = np.array([(i+1-CRPIX2)*CDELT2+CRVAL2 for i in range(NAXIS2)])/3600 #in deg
    [WL_pphi_edges, WL_tth_edges] = np.meshgrid(WL_phi_edges, WL_th_edges)
    
    #If latitude columns are full of nan, replace by zeros
    idx_nan = ~np.any(~np.isnan(WL_I_edges),axis=0)
    WL_I_edges[np.tile(idx_nan[:,np.newaxis].T,(NAXIS2,1))] = 0.
    
    #Also compute cell-centered grid
    WL_phi = (WL_phi_edges[1:]+WL_phi_edges[0:-1])/2
    WL_th = (WL_th_edges[1:]+WL_th_edges[0:-1])/2
    [WL_pphi, WL_tth] = np.meshgrid(WL_phi, WL_th)
    WL_I = (WL_I_edges[:-1,:-1]+WL_I_edges[1:,1:])/2
    
    return(WL_I,WL_pphi,WL_tth,WL_I_edges,WL_pphi_edges,WL_tth_edges)

import glob
def get_WL_map(WL_date,WL_path,WL_source,replace=False):
    sources = ['connect_tool',"V1.1"]
    if WL_source=='connect_tool':
            already_downloaded = glob.glob(f"{WL_path}/C2/connect_tool/*.png")
            WL_fullpath = None
            for filepath in already_downloaded :
                if WL_date.strftime("%Y%m%d") in os.path.basename(filepath)  :
                    WL_fullpath = filepath
                    break
            if WL_fullpath is None : 
                WL_fullpath = get_WL_map_connecttool(WL_date,WL_path)
    elif WL_source=='V1.1':
            [WL_fullpath,WL_date] = get_WL_map_local(WL_date,WL_path)
    else : raise ValueError(f"Source {WL_source} not in {sources}")
    return(WL_fullpath,WL_date)
            
# Fetch WL map from the public *connect_tool* web server
def get_WL_map_connecttool(WL_date,WL_path):
    url = "http://connect-tool.irap.omp.eu"
    sc = "ALL"
    helio = "PARKER"
    mag = "ADAPT"
    time = "SCTIME"
    try:
        #Interact with the web api to get download link
        url_api = url+"/api/"+sc+"/"+mag+"/"+helio+"/"+time+"/"+WL_date.strftime('%Y-%m-%d') + "/" + WL_date.strftime('%H%M%S')
        r = requests.get(url_api)
        txt = r.text
        pattern = '"click_to_download" href="'
        idx_start = txt.find(pattern) + len(pattern)
        idx_end = txt.find('.zip')
        url_static = url+txt[idx_start:idx_end+4]

        #Download the .zip collection file
        print("Fetching WL map from : "+ url_static)
        WL_path_tmp = os.path.join(WL_path,'C2','connect_tool')
        if not os.path.exists(WL_path_tmp) : os.makedirs(WL_path_tmp)
        zip_fullpath = wget.download(url_static,out=WL_path_tmp)

        #Extract WL map .png map from .zip archive
        base_filename = url_static.split('/')[-1][:-4]
        WL_filename = base_filename+"_backgroundwl.png"
        WL_fullpath = os.path.join(WL_path_tmp,WL_filename)
        with zipfile.ZipFile(zip_fullpath, 'r') as zip_ref:
            zip_ref.extract(WL_filename,path=WL_path_tmp)
    
        #Delete .zip archive
        os.remove(os.path.join(WL_path_tmp,base_filename+".zip"))
        
        return(WL_fullpath)
    except:
        raise Exception('Online archive: could not fetch WL map for input date: '+WL_date.strftime('%Y-%m-%dT%H:%M:%S'))

# Fetch WL map from the local database
def get_WL_map_local(WL_date,WL_path):
    WL_path_tmp = os.path.join(WL_path,'C2','V1.1(ISSI)')
    if not os.path.exists(WL_path_tmp) : os.makedirs(WL_path_tmp)
    files = []
    dates = []
    dates_diff = []
    for root, _, filenames in os.walk(WL_path_tmp):
        for filename in filenames:
            if ('_LC2_5p0Rs.fits' in filename): #e.g. 'WL_CRMAP_20181002T120000_LC2_5p0Rs.fits'
                current_date = datetime.strptime(filename.split('_')[2],'%Y%m%dT%H%M%S')
                dates.append(current_date) #Time is UTC
                files.append(os.path.join(root, filename))
                dates_diff.append(abs(current_date-WL_date))
    
    #Find the closest WL map available in time
    dates_diff = np.array(dates_diff)
    if np.any(dates_diff):
        idx = np.argmin(dates_diff)
        WL_fullpath = files[idx]
        WL_date_tmp = dates[idx]
        print('Local archive: the closest (in time) WL map available from input date: ' + WL_date.strftime('%Y-%m-%dT%H:%M:%S') + ' is: '+ str(WL_date_tmp))
        return(WL_fullpath,WL_date_tmp)
    else:
        raise Exception('Local archive: sorry, no WL map available for input date: ' + WL_date.strftime('%Y-%m-%dT%H:%M:%S'))