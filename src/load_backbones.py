import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import h5py
import utils
import sys

# SDXL unCLIP requires code from https://github.com/Stability-AI/generative-models/tree/main
sys.path.append('generative_models/')
import sgm
from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder # bigG embedder


# CONFIGS
data_path = "mindeyev2/"
num_voxels_list = []
hidden_dim = 1024 # 4096
seq_len = 1
subj_list = np.arange(2,9)
print(f"{subj_list=}")

n_blocks=4
blurry_recon=False # TODO check later
clip_scale=1

# get betas for subj01
# f = h5py.File(f'../{data_path}/betas_all_subj0{s}_fp32_renorm.hdf5', 'r')
# betas = f['betas'][:]
# # betas = torch.Tensor(betas).to("cpu").to(data_type)
# num_voxels_list.append(betas[0].shape[-1])
# num_voxels = betas[0].shape[-1]
# print(f"shape of betas: {betas.shape}")
# print(f"num_voxels for subj0{s}: {num_voxels}")

# get betas fro all subjects
num_voxels = {}
voxels = {}
for s in subj_list:

    f = h5py.File(f'../{data_path}/betas_all_subj0{s}_fp32_renorm.hdf5', 'r')
    betas = f['betas'][:]
    # betas = torch.Tensor(betas).to("cpu").to(data_type)
    num_voxels_list.append(betas[0].shape[-1])
    num_voxels[f'subj0{s}'] = betas[0].shape[-1]
    voxels[f'subj0{s}'] = betas
    print(f"num_voxels for subj0{s}: {num_voxels[f'subj0{s}']}")


# CLIP IMAGE EMBEDDINGS MODEL

def load_clip_model():

    clip_img_embedder = FrozenOpenCLIPImageEmbedder(
        arch="ViT-bigG-14",
        version="laion2b_s39b_b160k",
        output_tokens=True,
        only_tokens=True,
    )
    # clip_img_embedder.to(device)

    return clip_img_embedder

# CLIP MODEL PARAMS
clip_seq_dim = 256
clip_emb_dim = 1664


# MindEye MODEL CONTAINER

class MindEyeModule(nn.Module):
    def __init__(self):
        super(MindEyeModule, self).__init__()
    def forward(self, x):
        return x
        
model = MindEyeModule()

# RIDGE REGRESSION
# torch.Size([2, 1, 15724]) --> torch.Size([2, 1, 1024])

class RidgeRegression(torch.nn.Module):
    # make sure to add weight_decay when initializing optimizer
    def __init__(self, input_sizes, out_features, seq_len): 
        super(RidgeRegression, self).__init__()
        self.out_features = out_features
        self.linears = torch.nn.ModuleList([
                torch.nn.Linear(input_size, out_features) for input_size in input_sizes
            ])
    def forward(self, x, subj_idx):
        out = torch.cat([self.linears[subj_idx](x[:,seq]).unsqueeze(1) for seq in range(seq_len)], dim=1)
        return out
        
model.ridge = RidgeRegression(num_voxels_list, out_features=hidden_dim, seq_len=seq_len)

## Display # parameters in model
# utils.count_params(model.ridge)
# utils.count_params(model)

# test on subject 1 with fake data
b = torch.randn((2,seq_len,num_voxels_list[0]))
print(b.shape, model.ridge(b,0).shape)


# BRAIN MLP Network
# Input shape -- torch.Size([2, 1, 1024])
# Output shapes -- torch.Size([2, 256, 1664]) torch.Size([2, 256, 1664]) torch.Size([1]) torch.Size([1])

from models import BrainNetwork
model.backbone = BrainNetwork(h=hidden_dim, in_dim=hidden_dim, seq_len=seq_len, n_blocks=n_blocks,
                          clip_size=clip_emb_dim, out_dim=clip_emb_dim*clip_seq_dim, 
                          blurry_recon=blurry_recon, clip_scale=clip_scale)

## Displays # parameters in model
# utils.count_params(model.backbone)
# utils.count_params(model)

# test that the model works on some fake data
b = torch.randn((2,seq_len,hidden_dim))
print("b.shape",b.shape)

backbone_, clip_, blur_ = model.backbone(b)
print(backbone_.shape, clip_.shape, blur_[0].shape, blur_[1].shape)


# Load Pre-trained model
model_weights_path = "../mindeyev2/train_logs/multisubject_subj01_1024hid_nolow_300ep/last.pth"
model_state_dict = torch.load(model_weights_path)["model_state_dict"]
model.load_state_dict(model_state_dict, strict=False)


# Helper functions to import models in other scripts
def get_clip_image_model():
    return load_clip_model()

def get_mindeye_model():
    return model
