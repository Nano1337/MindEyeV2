import numpy as np
import torch
import webdataset as wds
import random
import h5py
from tqdm import tqdm

# LOAD PRETRAINED MODELS
from load_backbones import get_clip_image_model, get_mindeye_model
# clip_img_embedder = get_clip_image_model()
model = get_mindeye_model()

indicies_array = []

batch_size = 1
data_path = "../mindeyev2/"
s = 2
num_sessions = 40
num_voxels_list = []
train_data = {}
train_dl = {}
num_voxels = {}
voxels = {}
seq_len = 1

# def my_split_by_node(urls): return urls

# train_url = f"{data_path}/wds/subj0{s}/train/" + "{0.." + f"{num_sessions-1}" + "}.tar"
# print(f"{train_url=}")
# num_iterations_per_epoch = 5
# image_iters = torch.zeros(num_iterations_per_epoch, batch_size, 3, 224, 224).float()

# train_data = wds.WebDataset(train_url,resampled=True,nodesplitter=my_split_by_node)\
#                     .shuffle(750, initial=1500, rng=random.Random(42))\
#                     .decode("torch")\
#                     .rename(behav="behav.npy", past_behav="past_behav.npy", future_behav="future_behav.npy", olds_behav="olds_behav.npy")\
#                     .to_tuple(*["behav", "past_behav", "future_behav", "olds_behav"])
# train_dl = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False, drop_last=True)

# load in fMRI subject betas
f = h5py.File(f'{data_path}/betas_all_subj0{s}_fp32_renorm.hdf5', 'r')
betas = f['betas']# [:]
# betas = torch.Tensor(betas).to("cpu").to(torch.float16)
voxels = betas

path = "subj02_indices_array_10k.npy"
indicies_array = np.load(path)

embeddings = []
ct = 0
for img_idx, voxel_idx in tqdm(indicies_array, desc="Processing voxels"):  
    voxel0 = torch.from_numpy(voxels[voxel_idx])
    # print(f"orig {voxel0.shape=}")

    # unsqueeze channels
    voxel0 = voxel0.unsqueeze(1)
    # print(f"new {voxel0.shape=}")


    # generate voxel embeddings
    output = model.ridge(voxel0, s-2)
    # print(f"ridge {voxel_ridge.shape=}")

    # generate image CLIP embeddings
    # output = clip_img_embedder(image0.to(torch.float32)) # for image
    embeddings.append(output.cpu().detach().numpy())

embeddings_array = np.array(embeddings)
np.save("voxel_embeddings_10k.npy", embeddings_array)
