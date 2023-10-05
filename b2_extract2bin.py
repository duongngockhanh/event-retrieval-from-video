import numpy as np
import faiss
import glob
import torch

def save_bin(vectors, saved_file):
    faiss.normalize_L2(vectors)
    d = vectors.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(vectors)
    faiss.write_index(index, saved_file)

def process_and_save(first_path, saved_files):
    print(first_path)

    vectors = []
    second_path_list = sorted(glob.glob(f"{first_path}/*"))

    for second_path in second_path_list:
        third_path_list = sorted(glob.glob(f"{second_path}/*"))
        for third_path in third_path_list:
            vectors.append(np.load(third_path).reshape(-1, 512))
    vectors = np.concatenate(vectors, axis=0).astype(np.float32)
    save_bin(vectors, saved_files[0])
    print(vectors.shape) # (1038141, 512)

    v3_idx = np.array([i for i in range(vectors.shape[0]) if i % 3 != 1])
    vectors_v3 = vectors[v3_idx]
    save_bin(vectors_v3, saved_files[1])
    print(vectors_v3.shape)

    v5_idx = np.array([i for i in range(vectors.shape[0]) if i % 3 == 1])
    vectors_v5 = vectors[v5_idx]
    save_bin(vectors_v5, saved_files[2])
    print(vectors_v5.shape)

    print()

if __name__ == "__main__":
    first_path_v1 = "clip_features_Lxx"
    saved_files_v1 = ["keydata/full_faiss_v1.bin", "keydata/full_faiss_v3.bin", "keydata/full_faiss_v5.bin"]
    process_and_save(first_path_v1, saved_files_v1)

    first_path_v2 = "clip_features_Lxx_v2"
    saved_files_v2 = ["keydata/full_faiss_v2.bin", "keydata/full_faiss_v4.bin", "keydata/full_faiss_v6.bin"]
    process_and_save(first_path_v2, saved_files_v2)