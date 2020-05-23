# -*- coding: utf-8 -*-
"""
Experimental Vegetation Index Mapping program using DJI Mavic Pro DNG (raw image) +
JPEG 16-bit combo images taken using InfraBlue Filter 

This Version uses a special Broad Dual Bandpass NIR and Blue filter with a Narrow Bandpass in the Green
to Create a combo NIR Blue + Green Image Allowing for Standalone JPG to NDVI

%(c)-J. Campbell MuonRay Enterprises 2019 
This Python script was created using the Spyder Editor

"""
import warnings
warnings.filterwarnings('ignore')
from scipy import misc
import imageio
import numpy as np
from matplotlib import pyplot as plt  # For image viewing

#!/usr/bin/python
import getopt
import sys

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import ticker
from matplotlib.colors import LinearSegmentedColormap


class NDVI(object):
    def __init__(self, file_path, output_file=False, colors=False):
        self.image = plt.imread(file_path)
        self.output_name = 'NDVI.jpg'
        self.colors = ['gray', 'gray', 'red', 'yellow', 'green']

    def create_colormap(self, *args):
        return LinearSegmentedColormap.from_list(name='custom1', colors=args)

    def create_colorbar(self, fig, image):
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

    def convert(self):
        """
        This function performs the NDVI calculation and returns an GrayScaled frame with mapped colors)
        """
        NIR = (self.image[:, :, 0]).astype('float')
        blue = (self.image[:, :, 2]).astype('float')
        green = (self.image[:, :, 1]).astype('float')
        bottom = (blue - green) ** 2
        bottom[bottom == 0] = 1  # remove 0 from nd.array
        VIS = (blue + green) ** 2 / bottom
        NDVI = (NIR - VIS) / (NIR + VIS)

        fig, ax = plt.subplots()
        image = ax.imshow(NDVI, cmap=self.create_colormap(*self.colors))
        plt.axis('off')

        self.create_colorbar(fig, image)

        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(self.output_name, dpi=600, transparent=True, bbox_inches=extent, pad_inches=0)
        # plt.show()


def main(argv):
  
  
    input_file = 'DJI_0069.JPG'
    
    output_file = False
    colors = False

    blue_ndvi = NDVI(input_file, output_file=output_file or False, colors=colors or False)
    blue_ndvi.convert()


if __name__ == "__main__":
    main(sys.argv[1:])