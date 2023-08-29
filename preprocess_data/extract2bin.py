import numpy as np
import faiss
import glob

file_paths = glob.glob("clip-features-vit-b32/*.npy")

# Load your .npy files and concatenate vectors
vectors = []
for file_path in file_paths:  # Replace with your actual file paths
    vectors.append(np.load(file_path))
vectors = np.concatenate(vectors, axis=0).astype(np.float32)

# Normalize vectors (optional, but can improve search quality)
faiss.normalize_L2(vectors)

# Specify index type and dimensionality
d = vectors.shape[1]  # Dimensionality of vectors 
index = faiss.IndexFlatL2(d)  # L2 distance metric

# Add vectors to the index
index.add(vectors)

# Save the index and vectors to a binary file
faiss.write_index(index, "index_normal.bin")
np.savetxt("vectors_normal.bin", vectors)  # Save vectors as well