import torch
import webdataset as wds
import random
import h5py
from tqdm import tqdm
import numpy as np

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
num_iterations_per_epoch = 10000
image_iters = torch.zeros(num_iterations_per_epoch, batch_size, 3, 224, 224).float()

train_data = wds.WebDataset(train_url,resampled=True,nodesplitter=my_split_by_node)\
                    .shuffle(750, initial=1500, rng=random.Random(42))\
                    .decode("torch")\
                    .rename(behav="behav.npy", past_behav="past_behav.npy", future_behav="future_behav.npy", olds_behav="olds_behav.npy")\
                    .to_tuple(*["behav", "past_behav", "future_behav", "olds_behav"])
train_dl = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False, drop_last=True)

# Load 73k NSD images
f = h5py.File(f'{data_path}/coco_images_224_float16.hdf5', 'r')
images = f['images']# [:] # if you go OOM you can remove the [:] so it isnt preloaded to cpu! (will require a few edits elsewhere tho)

# load in fMRI subject betas
f = h5py.File(f'{data_path}/betas_all_subj0{s}_fp32_renorm.hdf5', 'r')
betas = f['betas']# [:]
# betas = torch.Tensor(betas).to("cpu").to(torch.float16)
voxels = betas

batch = next(iter(train_dl))

# print("train_dl batch:", batch)
# print("\n\n")
# idx = batch[0].to(torch.long)
# print(idx)
# idx = idx[:,0,0]
# print(idx)
# exit()


for iter, (behav0, past_behav0, future_behav0, old_behav0) in tqdm(enumerate(train_dl)):
    indicies_array.append((behav0[:,0,0].cpu().long(), behav0[:,0,5].cpu().long()))

    # # image0 = torch.from_numpy(images[behav0[:,0,0].cpu().long()])
    # # image_iters[iter,s*batch_size:s*batch_size+batch_size] = image0
    # voxel0 = voxels[behav0[:,0,5].cpu().long()]
    # print(f"orig {voxel0.shape=}")

    # voxel0 = torch.Tensor(voxel0)
    # voxel0 = voxel0.unsqueeze(1)

    # reorder channels for broadcasting
    # image0 = image0.permute(1, 2, 0)
    # print(f"shapes: {voxel0.shape=}, {image0.shape=}")


    # voxel_iters[f"subj0{subj_list[s]}_iter{iter}"] = voxel0

    # generate image CLIP embeddings
    # clip_target = clip_img_embedder(image0.to(torch.float32)) # for image
    # print(f"shapes: {clip_target.shape=}")

    ## WORKS
    # voxel_ridge = model.ridge(voxel0, s-2) # for voxels    

    ## DOES NOT WORK
    # backbone, clip_voxels, blurry_image_enc_ = model.backbone(voxel_ridge)

    # clip_scale > 1
    # clip_voxels_norm = nn.functional.normalize(clip_voxels.flatten(1), dim=-1)
    # clip_image_norm = nn.functional.normalize(clip_target.flatten(1), dim=-1)

    # print(f"shapes: {clip_voxels_norm.shape=}, {clip_image_norm.shape=}")

    if iter >= num_iterations_per_epoch:
        break

indicies_array = np.array(indicies_array)
np.save(f"subj0{s}_indices_array_all.npy", indicies_array)

exit()

data_type = torch.float16
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
with torch.no_grad(), torch.cuda.amp.autocast(dtype=data_type): 
    for test_i, (behav, past_behav, future_behav, old_behav) in enumerate(train_dl):  
        # all test samples should be loaded per batch such that test_i should never exceed 0

        voxel = voxels[behav[:,0,5].cpu().long()]
        voxel = voxel.unsqueeze(1)
        image = torch.from_numpy(images[behav[:,0,0].cpu().long()])

        print("shapes", voxel.shape, image.shape)

        exit()


        loss=0.
                    
        test_indices = torch.arange(len(test_voxel))[:300] # TODO: figure this out
        voxel = test_voxel[test_indices].to(device)
        image = test_image[test_indices].to(device)
        print(len(image))
        exit()
        # assert len(image) == 300


        clip_target = clip_img_embedder(image.float())

        for rep in range(3):
            voxel_ridge = model.ridge(voxel[:,rep],0) # 0th index of subj_list
            backbone0, clip_voxels0, blurry_image_enc_ = model.backbone(voxel_ridge)
            if rep==0:
                clip_voxels = clip_voxels0
                backbone = backbone0
            else:
                clip_voxels += clip_voxels0
                backbone += backbone0
        clip_voxels /= 3
        backbone /= 3