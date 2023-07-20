from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt


def scaleMinMax(x):
    return((x - np.nanmin(x))/(np.nanmax(x) - np.nanmin(x)))
def scaleCCC(x):
    return((x - np.nanpercentile(x, 2))/(np.nanpercentile(x, 98) - np.nanpercentile(x,2)))
def scaleStd(x):
    return((x - (np.nanmean(x)-np.nanstd(x)))/((np.nanmean(x)+np.nanstd(x)) - (np.nanmean(x)-np.nanstd(x))))

# ds = gdal.Open(r"C:\Users\Rutuja Kshirsagar\Downloads\AndraPradesh_6_2022-03-09_2A (1).tif") #newly sent resized img 

ds = gdal.Open(r"C:\Users\sanyu\OneDrive\Pictures\Saved Pictures\1. AndraPradesh_1_2022-01-23_2A.tif")
r = ds.GetRasterBand(4).ReadAsArray()
g = ds.GetRasterBand(3).ReadAsArray()
b = ds.GetRasterBand(2).ReadAsArray()

ds = None

rMinMax = scaleMinMax(r)
gMinMax = scaleMinMax(g)
bMinMax = scaleMinMax(b)

rgbMinMax = np.dstack((rMinMax,gMinMax,bMinMax))
# plt.figure()
# plt.imshow(rgbMinMax)
# plt.show() #displays a dark image

rCCC = scaleCCC(r)
gCCC = scaleCCC(g)
bCCC = scaleCCC(b)

rgbCCC = np.dstack((rCCC,gCCC,bCCC))
# plt.figure()
# plt.imshow(rgbCCC)
# plt.show() #medium bright image

rStd = scaleStd(r)
gStd = scaleStd(g)
bStd = scaleStd(b)

rgbStd = np.dstack((rStd,gStd,bStd))

plt.figure()
plt.imshow(rgbStd)
plt.show() #bright image