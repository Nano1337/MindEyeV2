import numpy as np
import torch
import webdataset as wds
import random
import h5py
from tqdm import tqdm

# LOAD PRETRAINED MODELS
from load_backbones import get_clip_image_model, get_mindeye_model
clip_img_embedder = get_clip_image_model()
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

def my_split_by_node(urls): return urls

train_url = f"{data_path}/wds/subj0{s}/train/" + "{0.." + f"{num_sessions-1}" + "}.tar"
print(f"{train_url=}")
num_iterations_per_epoch = 5
image_iters = torch.zeros(num_iterations_per_epoch, batch_size, 3, 224, 224).float()

train_data = wds.WebDataset(train_url,resampled=True,nodesplitter=my_split_by_node)\
                    .shuffle(750, initial=1500, rng=random.Random(42))\
                    .decode("torch")\
                    .rename(behav="behav.npy", past_behav="past_behav.npy", future_behav="future_behav.npy", olds_behav="olds_behav.npy")\
                    .to_tuple(*["behav", "past_behav", "future_behav", "olds_behav"])
train_dl = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False, drop_last=True)

# Load 73k NSD images
f = h5py.File(f'{data_path}/coco_images_224_float16.hdf5', 'r')
images = f['images']#[:] # if you go OOM you can remove the [:] so it isnt preloaded to cpu! (will require a few edits elsewhere tho)
# print("train_dl batch:", batch)
# print("\n\n")
# idx = batch[0].to(torch.long)
# print(idx)
# idx = idx[:,0,0]
# print(idx)
# exit()

num = 10 # in thousands

path = f"subj02_indices_array_{num}k.npy"
indicies_array = np.load(path)

embeddings = []

batch_size = 32
batch_images = None


for idx, (img_idx, voxel_idx) in enumerate(tqdm(indicies_array, desc="Processing images")):  
    image0 = torch.from_numpy(images[img_idx])
    # reorder channels for broadcasting
    image0 = image0.squeeze()

    output = clip_img_embedder(image0.to(torch.float32))
    output = torch.mean(output, dim=1).detach().cpu().numpy()
    output = output.squeeze()
    embeddings.append(output)

embeddings_array = np.array(embeddings)
np.save(f"clip_image_embeddings_{num}k.npy", embeddings_array)
