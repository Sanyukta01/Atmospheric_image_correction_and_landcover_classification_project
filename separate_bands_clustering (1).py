# -*- coding: utf-8 -*-
"""
LandSurfaceClustering.py for CWPRS IMAGE

Landon Halloran 
07.03.2019 
www.ljsh.ca

Land use clustering using multi-band remote sensing data.

This is a rough first version. Modifications will need to be made in order to properly 
treat other datasets.

Demo data is Sentinel-2 data (bands 2, 3, 4, 5, 6, 7, 8, 11 & 12) at 10m resolution 
(some bands are upsampled) in PNG format, exported from L1C_T32TLT_A007284_20180729T103019.
"""

# import these modules:
import matplotlib.pyplot as plt
import matplotlib as mpl
from osgeo import gdal
import numpy as np
import pandas as pd
import imageio.v2 as imageio
import glob
import seaborn as sns; sns.set(style="ticks", color_codes=True)
from matplotlib.colors import ListedColormap

########## USER MUST DEFINE THESE ###########
# image_folder_name = r"C:\Users\Rutuja Kshirsagar\Desktop\sep_band" # subfolder where images are located
image_folder_name = r"C:\Users\Dell\Desktop\Btech_project" # subfolder where images are located

image_format = 'png' # format of image files (the exact suffix of the filenames)
band_names = ['B02','B03','B04','B05','B06','B07','B08','B11','B12'] # names of bands (in file names). should all have some length
Nsamples = 20000 #  number of random samples used to "train" k-means here (for faster execution)
NUMBER_OF_CLUSTERS = 5# the number of independent clusters for k-means
# colour_map = 'terrain' # cmap, see matplotlib.org/examples/color/colormaps_reference.html
cluster_colors = ['green','peru', 'white', 'blue',"cyan"] # specify colors for each cluster
cmap = mpl.colors.ListedColormap(cluster_colors) # create a custom colormap
#  make the clustered image

#############################################

# import images to dictionary:
images = dict();
for image_path in glob.glob(image_folder_name+'/*.'+image_format):
    print('reading ',image_path)
    temp = imageio.imread(image_path)
    temp = temp[:,:,].squeeze()
    images[image_path[37:40]] = temp # FOR DIFFERENT FILE NAMES, ADJUST THIS!
print('images have ', np.size(temp),' pixels each')


# make a 3D numpy array of data...
imagecube = np.zeros([images['B02'].shape[0],images['B02'].shape[1],np.size(band_names)])
for j in np.arange(np.size(band_names)):
    imagecube[:,:,j] = images[band_names[j]] # 
imagecube=imagecube/256 #  scaling to between 0 and 1

# display an RGB or false colour image
thefigsize = (10,8)# set figure size
#plt.figure(figsize=thefigsize)
#plt.imshow(imagecube[:,:,0:3])

# sample random subset of images
imagesamples = []
for i in range(Nsamples):
    xr=np.random.randint(0,imagecube.shape[1]-1)
    yr=np.random.randint(0,imagecube.shape[0]-1)
    imagesamples.append(imagecube[yr,xr,:])
# convert to pandas dataframe
imagessamplesDF=pd.DataFrame(imagesamples,columns = band_names)


# make pairs plot (each band vs. each band)
##################################
# seaborn_params_p = {'alpha': 0.15, 's': 20, 'edgecolor': 'k'}
#pp1=sns.pairplot(imagessamplesDF, plot_kws = seaborn_params_p)#, hist_kws=seaborn_params_h)

# fit kmeans to samples:
from sklearn.cluster import KMeans

KMmodel = KMeans(n_clusters=NUMBER_OF_CLUSTERS) 
KMmodel.fit(imagessamplesDF.values)#values
KM_train = list(KMmodel.predict(imagessamplesDF.values)) #values
i=0
for k in KM_train:
    KM_train[i] = str(k) 
    i=i+1
imagessamplesDF2=imagessamplesDF
imagessamplesDF2['group'] = KM_train
# # pair plots with clusters coloured:
# pp2=sns.pairplot(imagessamplesDF,vars=band_names, hue='group',plot_kws = seaborn_params_p)
# pp2._legend.remove()

#  make the clustered image
imageclustered=np.empty((imagecube.shape[0],imagecube.shape[1]))
i=0
for row in imagecube:
    temp = KMmodel.predict(row) 
    imageclustered[i,:]=temp
    i=i+1
# plot the map of the clustered data
plt.figure(figsize=thefigsize)
# plt.imshow(imageclustered, cmap=colour_map) 
plt.imshow(imageclustered, cmap) 
plt.show()

# from sklearn.metrics import silhouette_score

# # Convert the clustered image to a 1D array
# imageclustered_1d = imageclustered.flatten()

# # Calculate the silhouette score for the clusters
# silhouette_avg = silhouette_score(imagessamplesDF.values, imageclustered_1d)

# print("The average silhouette_score is :", silhouette_avg)

# def scaleMinMax(x):
#     return((x - np.nanmin(x))/(np.nanmax(x) - np.nanmin(x)))
# def scaleCCC(x):
#     return((x - np.nanpercentile(x, 2))/(np.nanpercentile(x, 98) - np.nanpercentile(x,2)))
# def scaleStd(x):
#     return((x - (np.nanmean(x)-np.nanstd(x)))/((np.nanmean(x)+np.nanstd(x)) - (np.nanmean(x)-np.nanstd(x))))

# ds = gdal.Open(r"C:\Users\Rutuja Kshirsagar\Pictures\CWPRS dataset images\2. AndraPradesh_2_2022-01-28_2A.tif")

# r = ds.GetRasterBand(4).ReadAsArray()
# g = ds.GetRasterBand(3).ReadAsArray()
# b = ds.GetRasterBand(2).ReadAsArray()

# ds = None

# rMinMax = scaleMinMax(r)
# gMinMax = scaleMinMax(g)
# bMinMax = scaleMinMax(b)

# rgbMinMax = np.dstack((rMinMax,gMinMax,bMinMax))
# # plt.figure()
# # plt.imshow(rgbMinMax)
# # plt.show() #displays a dark image

# rCCC = scaleCCC(r)
# gCCC = scaleCCC(g)
# bCCC = scaleCCC(b)

# rgbCCC = np.dstack((rCCC,gCCC,bCCC))
# # plt.figure()
# # plt.imshow(rgbCCC)
# # plt.show() #medium bright image

# rStd = scaleStd(r)
# gStd = scaleStd(g)
# bStd = scaleStd(b)

# rgbStd = np.dstack((rStd,gStd,bStd))

# plt.figure()
# plt.imshow(rgbStd)
# plt.show() #bright image