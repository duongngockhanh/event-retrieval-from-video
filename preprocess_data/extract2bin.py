import numpy as np
import faiss
import glob
import torch

vectors = []

first_path = "clip_features_Lxx_v2"
saved_file = "full_faiss_v2.bin"

second_path_list = sorted(glob.glob(f"{first_path}/*"))

for second_path in second_path_list:
    third_path_list = sorted(glob.glob(f"{second_path}/*"))
    for third_path in third_path_list:
        vectors.append(np.load(third_path).reshape(-1, 512))
vectors = np.concatenate(vectors, axis=0).astype(np.float32)

print(vectors.shape) # (607407, 512)

# Normalize vectors (optional, but can improve search quality)
faiss.normalize_L2(vectors)

# Specify index type and dimensionality
d = vectors.shape[1]  # Dimensionality of vectors 
index = faiss.IndexFlatL2(d)  # L2 distance metric

# Add vectors to the index
index.add(vectors)

# Save the index and vectors to a binary file
faiss.write_index(index, saved_file)
# np.savetxt("vectors_normal_02.bin", vectors)  # Save vectors as well