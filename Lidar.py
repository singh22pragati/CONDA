#!/usr/bin/env python
# coding: utf-8

# In[1]:


import whiteboxgui
whiteboxgui.show(tree=True)


# In[2]:


import whitebox
wbt = whitebox.WhiteboxTools()
i=r'D:\RMSI Work\Lidar Data\NEONDSSampleLiDARPointCloud.las'
output= r'D:\Python\lidar\tree.las'
wbt.filter_lidar_classes(i, output, exclude_cls='0-4, 6-18')


# In[3]:


import whitebox
wbt = whitebox.WhiteboxTools()
cl = r'D:\Python\lidar\tree.las'
wbt.filter_lidar_classes(i=r'D:\RMSI Work\Lidar Data\NEONDSSampleLiDARPointCloud.las', output=(cl), exclude_cls='0-4, 6-18')


# In[4]:


dsm = r'D:\Python\lidar\dsm.tiff'
wbt.lidar_digital_surface_model(
    i=r'D:\RMSI Work\Lidar Data\NEONDSSampleLiDARPointCloud.las', 
    output= dsm, 
    resolution=0.5, 
    radius=0.25,
    minz=None, 
    maxz=None, 
    max_triangle_edge_length=None, 
)


# In[5]:


import whitebox
wbt = whitebox.WhiteboxTools()
ground = r'D:\Python\lidar\ground.las'
wbt.filter_lidar_classes(i=r'D:\RMSI Work\Lidar Data\NEONDSSampleLiDARPointCloud.las', output=(ground), exclude_cls='0-1, 3-18')


# In[6]:


dem = r'D:\Python\lidar\dem.tiff'
wbt.lidar_digital_surface_model(
    i=ground, 
    output= dem, 
    resolution=0.5, 
    radius=0.25, 
    minz=None, 
    maxz=None, 
    max_triangle_edge_length=None, 
)


# In[7]:


height = r'D:\Python\lidar\h.tiff'
wbt.subtract(
    dsm, 
    dem, 
    output = height
)


# In[8]:


import rasterio
import numpy as np

ds = rasterio.open("D:/Python/lidar/h.tiff")
data =ds.read()
lista = data.copy()

lista[np.where(lista >=1)] = 1
lista[np.where((lista <1) & (lista >=-32768))] = 0
np.asarray(lista, dtype=int)
with rasterio.open("D:/Python/lidar/tree_canopy.tiff", 'w',
           driver=ds.driver,
           height=ds.height,
           width=ds.width,
           crs=ds.crs,
           count=ds.count,
           transform=ds.transform,
           dtype=data.dtype

 )as dst:
   dst.write(lista)


# In[9]:


t = r'D:\Python\lidar\tree_canopy.tiff'
tree_canopy= r'D:\Python\lidar\tree_canopy.shp'

wbt.raster_to_vector_polygons(
    t, 
    output=tree_canopy
)


# In[10]:


#import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.plot import show
from rasterio.plot import show_hist
import rasterstats
import matplotlib.pyplot as plt
import os
import numpy
import shapely
from shapely.geos import WKTReadingError


# In[11]:


import geopandas as gpd
canopy = gpd.read_file(r'D:\Python\lidar\tree_canopy.shp')


# In[12]:


height = rasterio.open(r'D:\Python\lidar\h.tiff')


# In[13]:


fig, (ax1,ax2) = plt.subplots(1,2, figsize = (10,4))
show(height, ax = ax1, title = 'Height')
canopy.plot(ax = ax1, facecolor = 'None', edgecolor = 'red')
show_hist(height, title = 'Histogram', ax = ax2)
plt.show


# In[14]:


hr = height.read(1)
affine = height.transform


# In[15]:


maxh = rasterstats.zonal_stats(canopy, hr, affine = affine, stats = ['max'], geojson_out = True)
Max_height = []


# In[16]:


i = 0


# In[17]:


while i < len(maxh):
    Max_height.append(maxh[i])
    i = i + 1


# In[18]:


maxh_canopy = pd.DataFrame(Max_height)


# In[19]:


print(maxh_canopy)
maxh_canopy.to_csv(r'D:\Python\lidar\tree_canopy_h.csv')


# In[20]:


import pandas as pd
import geopandas as gpd
data = gpd.read_file(r'D:\Python\lidar\tree_canopy.shp')
data['geometry'].representative_point()
data.to_file(r'D:\Python\lidar\tree_cany.shp')
data


# In[21]:


h = pd.merge(data, maxh_canopy, right_index=True, left_index = True)
h


# In[22]:


type(h)


# In[27]:


h.to_csv(r'D:\Python\lidar\tree_cany.csv')


# In[ ]:




