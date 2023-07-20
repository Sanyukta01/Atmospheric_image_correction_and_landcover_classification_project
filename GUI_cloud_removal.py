from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import os
import pandas as pd
import imageio
import glob
import seaborn as sns; sns.set(style="ticks", color_codes=True)
from sklearn.cluster import KMeans
from sklearn.metrics import homogeneity_score
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score
from sklearn.datasets import make_blobs
# Create the main window
root = Tk()
root.title("Project")
def select_file():
    global file_path
    file_path = filedialog.askopenfilename()
def scaleStd(x):
    return((x - (np.nanmean(x)-np.nanstd(x)))/((np.nanmean(x)+np.nanstd(x)) - (np.nanmean(x)-np.nanstd(x))))
# Define the functions for the three buttons
def display_image():
    # Perform operation 1 on the image    
    ds = gdal.Open(file_path)
    r = ds.GetRasterBand(4).ReadAsArray()
    g = ds.GetRasterBand(3).ReadAsArray()
    b = ds.GetRasterBand(2).ReadAsArray()
    ds = None
    rStd = scaleStd(r)
    gStd = scaleStd(g)
    bStd = scaleStd(b)
    rgbStd = np.dstack((rStd,gStd,bStd))
    plt.figure()
    plt.imshow(rgbStd)
    plt.show()
    print("The image has been displayed!!")

def resize_img():
    # Perform operation 2 on the image
    input_path = filedialog.askopenfilename(title="Select Input TIFF file", filetypes=[("TIFF files", "*.tif")])
    input_tiff = gdal.Open(input_path)
    ref_tiff = gdal.Open(file_path)
    # Get the reference image's height and width
    ref_height, ref_width = ref_tiff.RasterYSize, ref_tiff.RasterXSize
    print("Reference image size:", ref_height, "x", ref_width)
    # Get the number of bands in the input image
    num_bands = input_tiff.RasterCount
    print("Number of input image bands:", num_bands)
    # Get the data type of the input image
    data_type = input_tiff.GetRasterBand(1).DataType
    # Create a new output TIFF file with the same geotransform and projection as the input TIFF file
    driver = gdal.GetDriverByName('GTiff')
    output_tiff = driver.Create(r'C:\Users\hp\Desktop\New folder\output_tiff.tif', ref_width, ref_height, num_bands, data_type)
    output_tiff.SetProjection(ref_tiff.GetProjection())
    output_tiff.SetGeoTransform(ref_tiff.GetGeoTransform())
    # Resize the input image and write it to the output image
    for i in range(num_bands):
        input_band = input_tiff.GetRasterBand(i+1)
        output_band = output_tiff.GetRasterBand(i+1)
        gdal.ReprojectImage(input_tiff, output_tiff, None, None, gdal.GRA_Bilinear)
    # Close the input and output files
    input_tiff = None
    output_tiff = None
    # Open the image
    with rasterio.open(r'C:\Users\hp\Desktop\New folder\output_tiff.tif') as src:
        # Get the size of the image
        width, height = src.width, src.height
        # Print the size
        print("The size of the output resized image is:", height, "x", width)
    print("Resizing Done!")

def cloud_free():
    # Perform operation 3 on the image
# Open the cloud cover image
  with rasterio.open(file_path) as cloud_src:
    cloud_image = cloud_src.read()
# Open the cloud-free image
#resize output
    with rasterio.open(r'C:\Users\hp\Desktop\New folder\output_tiff.tif') as clear_src:
     clear_image = clear_src.read()
# Create a binary cloud mask from the cloud image
    threshold = 12500
    cloud_mask = np.zeros_like(cloud_image)
    cloud_mask[cloud_image < threshold] = 1
# Replace only the pixels corresponding to clouds in the cloud image with the corresponding pixels from the clear image
    reconstructed_image = np.copy(cloud_image)
    reconstructed_image[cloud_mask == 1] = clear_image[cloud_mask == 1]
   # Write the reconstructed image to a new file
    meta = cloud_src.meta
    meta.update(dtype=rasterio.float32, count=clear_image.shape[0])
    with rasterio.open(r'C:\Users\hp\Desktop\New folder\reconstructed_image.tif', "w", **meta) as dest:
     dest.write(reconstructed_image.astype(rasterio.float32))
    ds = gdal.Open(r'C:\Users\hp\Desktop\New folder\reconstructed_image.tif')
    r = ds.GetRasterBand(4).ReadAsArray()
    g = ds.GetRasterBand(3).ReadAsArray()
    b = ds.GetRasterBand(2).ReadAsArray()
    ds = None
    rStd = scaleStd(r)
    gStd = scaleStd(g)
    bStd = scaleStd(b)
    rgbStd = np.dstack((rStd,gStd,bStd))
    plt.figure()
    plt.imshow(rgbStd)
    plt.show()   
    print("The clouds are removed!!")

def k_clustering():
   src_ds = gdal.Open(file_path)
   # Get the number of bands in the image
   num_bands = src_ds.RasterCount
  # Loop through each band
   band_folder=r"C:\Users\hp\Desktop\New folder"
   for i in range(num_bands):
    # Get the band data
     band = src_ds.GetRasterBand(i+1)
     data = band.ReadAsArray()
     data=data*2 #brighten image
     # Saving each band as different tiff file
     driver = gdal.GetDriverByName("GTiff")
     if i<9:
        output_path = os.path.join(band_folder,f"_B0{i+1}_.png")
     else:
        output_path = os.path.join(band_folder,f"_B{i+1}_.png")
     out_ds = driver.Create(output_path, src_ds.RasterXSize, src_ds.RasterYSize, 1, band.DataType)
     out_ds.GetRasterBand(1).WriteArray(data)
     out_ds = None
     print(f'Band {i+1} saved at: {output_path}')
   image_folder_name = band_folder
   image_format = 'png' # format of image files (the exact suffix of the filenames)
   band_names = ['B01','B02','B03','B04','B05','B06','B07','B08','B11','B12'] # names of bands (in file names). should all have some length
   Nsamples = 20000 #  number of random samples used to "train" k-means here (for faster execution)
   NUMBER_OF_CLUSTERS = 5 # the number of independent clusters for k-means
   colour_map = 'terrain' # cmap, see matplotlib.org/examples/color/colormaps_reference.html
        # import images to dictionary:
   images = dict();
   for image_path in glob.glob(image_folder_name+'/*.'+image_format):
       print('reading ',image_path)
       temp = imageio.v2.imread(image_path)
       temp = temp[:,:,].squeeze()
       images[image_path[32:35]] = temp # FOR DIFFERENT FILE NAMES, ADJUST THIS!
   print('images have ', np.size(temp),' pixels each')
   print(images.keys())
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
   #seaborn_params_p = {'alpha': 0.15, 's': 20, 'edgecolor': 'k'}
#pp1=sns.pairplot(imagessamplesDF, plot_kws = seaborn_params_p)#, hist_kws=seaborn_params_h)
# fit kmeans to samples:
   KMmodel = KMeans(n_clusters=NUMBER_OF_CLUSTERS) 
   KMmodel.fit(imagessamplesDF.values)#values
   KM_train = list(KMmodel.predict(imagessamplesDF.values)) #values
   i=0
   for k in KM_train:
       KM_train[i] = str(k) 
       i=i+1
   imagessamplesDF2=imagessamplesDF
   imagessamplesDF2['group'] = KM_train
   # pair plots with clusters coloured:
   #pp2=sns.pairplot(imagessamplesDF,vars=band_names, hue='group',plot_kws = seaborn_params_p)
   #pp2._legend.remove()
   #  make the clustered image
   imageclustered=np.empty((imagecube.shape[0],imagecube.shape[1]))
   i=0
   wcss=[]
   for row in imagecube:
       temp = KMmodel.predict(row) 
       imageclustered[i,:]=temp
       i=i+1
# plot the map of the clustered data
   plt.figure(figsize=thefigsize)
   plt.imshow(imageclustered, cmap=colour_map) 
   plt.show()
#for i in range(1, 11):
 #   kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
  #  kmeans.fit(imagessamplesDF.values)
   # print(kmeans.inertia_)
    #wcss.append(kmeans.inertia_)
#sse = KMmodel.inertia_
#number_clusters = range(1,11)
#plt.plot(number_clusters,wcss)
#plt.title('The Elbow title')
#plt.xlabel('Number of clusters')
#plt.ylabel('WCSS')
#plt.show()
#print(sse)
# calculate homogeneity score
   homogeneity = homogeneity_score(KM_train, KMmodel.labels_)
   print('Homogeneity score:', homogeneity)
   i=0
   for k in KM_train:
       KM_train[i] = str(k) 
       i=i+1
   imagessamplesDF2=imagessamplesDF
   imagessamplesDF2['group'] = KM_train
#  make the clustered image
   imageclustered=np.empty((imagecube.shape[0],imagecube.shape[1]))
   i=0
   wcss=[]
   for row in imagecube:
       temp = KMmodel.predict(row) 
       imageclustered[i,:]=temp
       i=i+1
# Compute Silhouette Coefficient
   silhouette = silhouette_score(imagessamplesDF.values, KMmodel.labels_)
   print('Silhouette score: ',silhouette)
   # Compute Davies-Bouldin Index
   davies_bouldin = davies_bouldin_score(imagessamplesDF.values, KMmodel.labels_)
   print('Davies Bouldin index: ',davies_bouldin)
# Compute Calinski-Harabasz Index
   calinski_harabasz = calinski_harabasz_score(imagessamplesDF.values, KMmodel.labels_)
   print('Calinski Harabasz index: ',calinski_harabasz)  
   print("The image has been clustered!!")

# Create the three buttons
# Select file button
select_file_button = Button(root, text="Select File", command=select_file)
input_label = Label(root, text="Welcome to Atmospheric Correction & Classification !!")
input_label.pack()
button_2 = Button(root, text="Resize", command=resize_img)
button_1 = Button(root, text="Display", command=display_image)
button_3 = Button(root, text="Remove Clouds", command=cloud_free)
button_4 = Button(root, text="Cluster the Image", command=k_clustering)
# Add the buttons to the window
select_file_button.pack(pady=10)
button_1.pack(pady=10)
button_2.pack(pady=10)
button_3.pack(pady=10)
button_4.pack(pady=10)
# Run the GUI loop
root.mainloop()