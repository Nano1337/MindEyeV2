import numpy as np

dim = 5
B = 32
# Assuming 'embeddings' is your list of (1, dim) numpy elements
embeddings = [np.random.rand(dim) for _ in range(B)]  # Example list

embeddings = np.array(embeddings)
print(embeddings.shape)  # (32, 1, 5)