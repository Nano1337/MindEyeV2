import h5py
import nibabel as nib
import numpy as np
import vedo
import matplotlib.pyplot as plt
import torch

# load in fMRI subject betas
data_path = "../mindeyev2/"
s = 2
f = h5py.File(f'{data_path}/betas_all_subj0{s}_fp32_renorm.hdf5', 'r')
betas = f['betas']# [:]

# betas = torch.Tensor(betas).to("cpu").to(torch.float16)
voxels = betas

idx_path = "subj02_indices_array_10k.npy"
indicies_array = np.load(idx_path)

# num_voxels in subj02 == 14278
# dim of betas for subj02 == (146, 190, 150, 750)
# dim of time_series for subj02 == (146, 190, 150, 301)

for img_idx, voxel_idx in indicies_array:  
  voxel0 = torch.from_numpy(voxels[voxel_idx])
  voxel0 = voxel0.unsqueeze(1)
  voxel0 = voxel0.cpu().detach().numpy()

  # print(voxel0.shape)
  # print(type(voxel0[0,0,0]))
  # exit(0)

  # reformat data into 4D tensor
  # 118*121 = 14278
  data = voxel0.reshape(118,121,1,1)

  new_image = nib.Nifti1Image(data, affine=np.eye(4), dtype=np.float32)

  # Save the Nifti image to a file
  nib.save(new_image, "my_image.nii.gz")

  exit(0)

# voxel embeddings conversion
# size (1, 1, 1024) where 32*32 = 1024

path = "voxel_embeddings_1k.npy"
emb_array = np.load(path)

for emb in emb_array:
  # reformat data into 4D tensor
  data = emb.reshape(32,32,1,1)

  new_image = nib.Nifti1Image(data, affine=np.eye(4), dtype=np.float32)
  # print(new_image)

  # Save the Nifti image to a file
  nib.save(new_image, "my_image.nii.gz")

  exit(0)

# Access data using get_fdata()
# volume_data = new_image.get_fdata()
# print(volume_data.shape)

# IGNORE BELOW
# CHECK VISUALIZE_NIFTY.PY

# # Define titles for each subplot based on axis
# slice_titles = ['X-axis Average', 'Y-axis Average', 'Z-axis Average']

# # Create a figure and subplots with 1 row and 3 columns
# fig, axes = plt.subplots(1, 3, figsize=(12, 4))  # Adjust figsize for better visualization

# # Loop through each axis (0 for X, 1 for Y, 2 for Z)
# for i, axis in enumerate([0, 1, 2]):
#   # Average data along the chosen axis
#   averaged_slice = np.mean(volume_data, axis=axis)
#   averaged_slice = averaged_slice  # scale for float range [0,1]

#   # Create the subplot for the current slice
#   if not i:
#     im = axes[i].imshow(averaged_slice.T, cmap='gray', origin='lower')
#   else:
#     axes[i].imshow(averaged_slice.T, cmap='gray', origin='lower')
#   axes[i].set_title(slice_titles[i])
#   axes[i].set_xlabel('X axis')
#   axes[i].set_ylabel('Y axis')
#   axes[i].set_xticks([])  # Optional: Remove x-axis ticks for clarity
#   axes[i].set_yticks([])  # Optional: Remove y-axis ticks for clarity

#   fig.colorbar(im, ax=axes[i], label='Signal intensity')  # Use the plotted image data (im)

# Add a single colorbar for all subplots (optional)
# fig.colorbar(plt.cm.ScalarMappable(cmap='gray'), label='Signal intensity', fraction=0.04, pad=0.02)  # Adjust position with fraction and pad

# Save the plot as an image file
# plt.savefig('averaged_slices.png')
# plt.show()


# z_slice = volume_data[ :, :, 0, 0]
# plt.imshow(z_slice.T, cmap='gray', origin='lower')
# plt.xlabel('X axis')
# plt.ylabel('Y axis')
# plt.colorbar(label='Signal intensity')

# # Save the plot as an image file
# plt.savefig('my_slice_image.png')


# # Extract dimensions from affine matrix (assuming a standard voxel space)
# # Modify indexing if your affine represents a different space
# dims = new_image.affine[:3, :3].diagonal()
# dims = [int(dim) for dim in dims]

# # Convert nibabel image to vedo volume object
# vol = vedo.Volume(volume_data, dims)

# # Customize visualization (optional)
# vol.cmap('jet')  # Set colormap
# # vol.lighting(ambient=(0.2, 0.2, 0.2), directional=(0.8, 0.5, 0.3))  # Adjust lighting

# # Show the visualization
# vedo.show(vol)