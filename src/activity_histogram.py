import numpy as np
import torch
import matplotlib.pyplot as plt


def min_max_scale(data):
  """
  Scales data between 0 and 1 using min-max scaling.
  """
  min_val = np.min(data)
  max_val = np.max(data)
  return (data - min_val) / (max_val - min_val)

def plot_histogram(data, title="fMRI Betas Distribution", x_label="Normalized Activation", 
                   y_label="Frequency", alpha=0.4, bin_color="blue"):
  """
  Creates a histogram with 20 bins for values between 0 and 1,
  formats the plot, and displays it using matplotlib.
  """
  plt.hist(data, bins=20, range=(0, 1), color=(bin_color, alpha))
  plt.title(title)
  plt.xlabel(x_label)
  plt.ylabel(y_label)

path = "voxel_embeddings_10k.npy"
data_array = np.load(path)

idx = 150
data = data_array[idx]

embedding = data.squeeze()
    
scaled_emb = min_max_scale(embedding)

plot_histogram(scaled_emb)

savepath = "activity_hist.png"
plt.savefig(savepath)
