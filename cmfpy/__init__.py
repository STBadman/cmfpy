# %%
from cmfpy.model import model
from cmfpy.utils import listhtml
from numpy import save
import os

GONG_URL = f'https://gong.nso.edu/adapt/maps/gong/'
GONG_YEARS = [2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023]
__all__ = ['model']
__model_cachepath__ = f'{__path__[0]}/model/__cache__'
if not os.path.exists(__model_cachepath__): os.makedirs(__model_cachepath__)

def cache_magneo_urls():
    for year in GONG_YEARS:
        url = f'https://gong.nso.edu/adapt/maps/gong/{year}/'
        files = listhtml(url, contains=f'{year}', include_url=False)
        save(f'{__model_cachepath__}/magneto_urls_{year}.npy',files,allow_pickle=True)
#cache_magneo_urls()
# %%
