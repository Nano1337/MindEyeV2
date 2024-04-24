import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# load fMRI embeddings
voxels = np.load('voxel_embeddings_10k.npy')
voxels = voxels.squeeze()

# load cluster labels
df = pd.read_csv('clustering_results_10k_euclidean.csv')
labels = df['cluster_label'].to_numpy()

# filter from labels
not_minus_one_or_zero = (labels != -1) & (labels != 0) # bit-wise
indices = np.flatnonzero(not_minus_one_or_zero)
labels = labels[indices]

# 80/20 random split
X_train, X_test, y_train, y_test = train_test_split(voxels, labels, test_size=0.2, random_state=42)

# one-hot encoding of labels
encoder = OneHotEncoder()
y_train_encoded = encoder.fit_transform(y_train.reshape(-1, 1))

print(f"{y_train_encoded.shape=}")
print(set(y_train))
exit(0)

# define and train various classification models
models = {}

# logistic regression
models["Logistic Regression"] = LogisticRegression()
models["Logistic Regression"].fit(X_train, y_train_encoded)

# svm classifier
models["SVM"] = SVC()
models["SVM"].fit(X_train, y_train_encoded)

# evaluate performance
for model_name, model in models.items():
  
  # predictions
  y_pred = model.predict(X_test)

  # compute accuracy
  accuracy = accuracy_score(y_test, y_pred)
  print(f"{model_name} Accuracy: {accuracy:.4f}")