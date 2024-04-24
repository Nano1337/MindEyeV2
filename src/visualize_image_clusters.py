import pandas as pd
import h5py
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import numpy as np

def visualize_and_save_images(images, save_path='cluster_visualization_output.png'):
    """
    Visualizes the first 25 images in a 5x5 grid and saves the plot as a PNG file.

    Parameters:
    - images: A batch of images as a numpy array with shape (N, C, H, W).
    - save_path: Path to save the output PNG file.
    """

    images = images.astype(np.float32)

    if images.shape[0] < 25:
        raise ValueError("The batch of images must contain at least 25 images.")

    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        # Transpose the image from (C, H, W) to (H, W, C) for visualization
        img = images[i].transpose(1, 2, 0)
        # Assuming the images are stored as float values in [0, 1], if not, you might need to adjust.
        plt.imshow(img, cmap=plt.cm.binary)

    plt.savefig(save_path)
    plt.close()  # Close the figure to free memory

if __name__ == "__main__":

    num=10
    path = f"subj02_indices_array_{num}k.npy"
    indicies_array = np.load(path)

    data_path = "../mindeyev2/"

    # Load the clustering results from the CSV file
    file_path = 'clustering_results_10k.csv'
    df = pd.read_csv(file_path)

    # Specify the cluster label you want to visualize
    cluster_label = 0

    # Filter the DataFrame for the specified cluster label
    filtered_df = df[df['cluster_label'] == cluster_label]

    # Extract and print the indices of the filtered DataFrame
    indices = filtered_df['index'].tolist()
    print(f"Indices for cluster label {cluster_label}: {indices}")

    f = h5py.File(f'{data_path}/coco_images_224_float16.hdf5', 'r')
    images = f['images']#[:] # if you go OOM you can remove the [:] so it isnt preloaded to cpu! (will require a few edits elsewhere tho)

    img_indices = indicies_array[indices][:, 0]
    retrieved_images = []
    for img_index in img_indices:
        img = images[img_index][0, :, :, :]
        retrieved_images.append(img)

    retrieved_images = np.array(retrieved_images)
    print(f"Retrieved images: {retrieved_images.shape}")

    # Check if the batch size is less than 25
    if retrieved_images.shape[0] < 25:
        # Calculate how many images are needed to pad the batch to 25
        num_images_to_pad = 25 - retrieved_images.shape[0]
        # Create an array of zeros with the shape of one image
        padding = np.zeros((num_images_to_pad, *retrieved_images.shape[1:]), dtype=retrieved_images.dtype)
        # Pad the retrieved_images array
        retrieved_images = np.concatenate((retrieved_images, padding), axis=0)

    # Visualize and save the retrieved images
    visualize_and_save_images(retrieved_images, save_path='cluster_visualization_output.png')