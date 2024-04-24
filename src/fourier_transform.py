import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

def plot_fft_and_save(data, filename, x_label="Time", y_label="Signal", title="Original Signal"):
  """
  Plots the Fast Fourier Transform (FFT) magnitude spectrum of a 1D data array 
  and saves it as an image file.

  Args:
      data (np.array): The 1D data array.
      filename (str): The filename (including extension) to save the image.
      x_label (str, optional): Label for the x-axis (default: "Time").
      y_label (str, optional): Label for the y-axis (default: "Signal").
      title (str, optional): Title for the plot (default: "Original Signal").
  """
  
  x = np.linspace(0.0, 1.0, len(data))  # Time axis (adjust for your data)

  yf = fft(data)  # Compute FFT of the signal

  xf = fftfreq(len(data), d=(x[1] - x[0]))  # Frequencies corresponding to FFT bins

  magnitude_spectrum = 2.0/len(data) * np.abs(yf)  # One-sided magnitude spectrum

  plt.figure(figsize=(8, 5))  # Set plot size

  plt.subplot(2, 1, 1)  # Top subplot for original data
  plt.plot(x, data)
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.title(title)
  plt.grid(True)

  plt.subplot(2, 1, 2)  # Bottom subplot for magnitude spectrum
  plt.plot(xf, magnitude_spectrum)
  plt.xlabel('Frequency (Hz)')
  plt.ylabel('Magnitude')
  plt.title('Magnitude Spectrum')
  plt.grid(True)

  plt.tight_layout()  # Adjust spacing between subplots

  plt.savefig(filename)  # Save the plot as an image
  plt.close()  # Close the plot window


# LOAD in data from np.array
path = "voxel_embeddings_1k.npy"
data_array = np.load(path)

# SAMPLE from data
data = data_array[0].squeeze()

# FT PLOT
plot_fft_and_save(data, "fft_spectrum.png", x_label="Time (s)", y_label="Amplitude", title="Sample Signal")

print("Magnitude spectrum saved as fft_spectrum.png")
