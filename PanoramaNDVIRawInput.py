# -*- coding: utf-8 -*-
"""
Created on Mon May 25 18:51:53 2020

@author: cosmi
"""

#Image stitching (Panorama Maker DNG Batch Version)"


import os
import rawpy
import imageio
import cv2
import numpy as np

## Image Processing libraries
from skimage import exposure

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import ticker
from matplotlib.colors import LinearSegmentedColormap


images=[]
for infile in os.listdir("./"):
    print( "file : " + infile)
    if infile[-3:] == "tif" or infile[-3:] == "DNG" :
       # print "is tif or DNG (RAW)"
       outfile = infile[:-3] + "jpg"
       
       print( "new filename : " + outfile)
       dim=(640,360)
       raw = rawpy.imread(infile)
       # Postprocessing, i.e demosaicing here, will always 
#change the original pixel values. Typically what you want
# is to get a linearly postprocessed image so that roughly 
#the number of photons are in linear relation to the pixel values. 
#You can do that with:

       rgb = raw.postprocess()
      
       rgb = cv2.resize(rgb,dim,interpolation = cv2.INTER_AREA)
       rgb = cv2.bitwise_not(~rgb)


       images.append(rgb)
       #Read the images from your directory
       


#stitcher = cv2.createStitcher()
stitcher = cv2.Stitcher.create()
ret,pano = stitcher.stitch(images)

if ret==cv2.STITCHER_OK:
    
    #need to swap colors here, demosaicing algoithm in rawpy jumbles to colors for some reason?:
    truepano = cv2.cvtColor(pano,cv2.COLOR_BGR2RGB)
    
    #Apply gamma corrections: gamma values greater than 1 will shift the image histogram towards left and the output image will be darker than the input image. On the other hand, for gamma values less than 1, the histogram will shift towards right and the output image will be brighter than the input image.
    

    gamma_corrected_pano = exposure.adjust_gamma(truepano, gamma=1, gain=0.5)

       
    panoimage=gamma_corrected_pano
       
    #apply histogram equalization
    #using skimage (easy way)
    hist_equalized_pano = exposure.equalize_hist(panoimage)
    
    
    panoramaoutfile = "panoresult.png"

    
    imageio.imsave(panoramaoutfile, pano)


    cv2.imshow('Panorama',hist_equalized_pano)
    cv2.waitKey()
    cv2.destroyAllWindows()
else: 
    print("Error during Stitching")


# Open an image
image = pano

# Get the red band from the rgb image, and open it as a numpy matrix
#NIR = image[:, :, 0]
         
#ir = np.asarray(NIR, float)


ir = (image[:,:,0]).astype('float')


# Get one of the IR image bands (all bands should be same)
#blue = image[:, :, 2]

#r = np.asarray(blue, float)

r = (image[:,:,2]).astype('float')


# Create a numpy matrix of zeros to hold the calculated NDVI values for each pixel
ndvi = np.zeros(r.size)  # The NDVI image will be the same size as the input image

# Calculate NDVI
ndvi = np.true_divide(np.subtract(ir, r), np.add(ir, r))


# Display the results
output_name = 'InfraBlueNDVI3.jpg'

#a nice selection of grayscale colour palettes
cols1 = ['blue', 'green', 'yellow', 'red']
cols2 =  ['gray', 'gray', 'red', 'yellow', 'green']
cols3 = ['gray', 'blue', 'green', 'yellow', 'red']

cols4 = ['black', 'gray', 'blue', 'green', 'yellow', 'red']

def create_colormap(args):
    return LinearSegmentedColormap.from_list(name='custom1', colors=cols3)

#colour bar to match grayscale units
def create_colorbar(fig, image):
        position = fig.add_axes([0.125, 0.19, 0.2, 0.05])
        norm = colors.Normalize(vmin=-1., vmax=1.)
        cbar = plt.colorbar(image,
                            cax=position,
                            orientation='horizontal',
                            norm=norm)
        cbar.ax.tick_params(labelsize=6)
        tick_locator = ticker.MaxNLocator(nbins=3)
        cbar.locator = tick_locator
        cbar.update_ticks()
        cbar.set_label("NDVI", fontsize=10, x=0.5, y=0.5, labelpad=-25)

fig, ax = plt.subplots()
image = ax.imshow(ndvi, cmap=create_colormap(colors))
plt.axis('off')

create_colorbar(fig, image)

extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.savefig(output_name, dpi=600, transparent=True, bbox_inches=extent, pad_inches=0)
        # plt.show()

 