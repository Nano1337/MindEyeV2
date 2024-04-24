# SOURCE: https://peerherholz.github.io/workshop_weizmann/data/image_manipulation_nibabel.html

# import settings
import sys
from nilearn import plotting
import pylab as plt

import numpy as np
import nibabel as nb

# Load functional image
filepath = 'raw_nsd/betas_session01.nii.gz'
if len(sys.argv) > 1: 
    filepath = sys.argv[1]
img = nb.load(filepath)

# img = nb.load('raw_nsd/timeseries_session01_run02.nii.gz')

# Display file header
# print(img)

data = img.get_fdata()
# print(data.shape) # returns (X, Y, Z, 1)

affine = img.affine
# print(affine) # transformation matrix

header = img.header['pixdim']
# print(header) # voxel resolution

# print((data.dtype, img.get_data_dtype()))
# returns (dtype('float64'), dtype('<f4'))

# save path
savepath = 'my_slice_image_ortho1.png'
if len(sys.argv) > 2:
    savepath = sys.argv[2]

# DISPLAY Z Slice
# plt.imshow(data[:, :, data.shape[2] // 2, 0].T, cmap='Greys_r')
# plt.savefig(savepath)
# print(data.shape)

img.orthoview()
plt.savefig(savepath)

