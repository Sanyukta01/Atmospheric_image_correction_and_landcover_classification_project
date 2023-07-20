import numpy as np
import rasterio

# Open the cloud cover image
with rasterio.open(r"C:\Users\Dell\Pictures\CWPRS dataset images\1. AndraPradesh_1_2022-01-23_2A (1).tif") as cloud_src:
    cloud_image = cloud_src.read()

# Open the cloud-free image
#resize output
with rasterio.open(r"C:\Users\Dell\output_tiff.tif") as clear_src:
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
with rasterio.open(r"C:\Users\Dell\Desktop\Btech_project\reconstructed_image.tif", "w", **meta) as dest:
    dest.write(reconstructed_image.astype(rasterio.float32))
