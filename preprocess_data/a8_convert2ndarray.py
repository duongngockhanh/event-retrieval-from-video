import numpy as np
import torch
import glob

first_path = "clip_features_Lxx_v2"
second_path_list = glob.glob(f"{first_path}/*")
for second_path in second_path_list:
    third_path_list = glob.glob(f"{second_path}/*")
    for third_path in third_path_list:
        temp = np.load(third_path, allow_pickle=True)
        if not isinstance(temp[0], np.ndarray):
            temp = np.array([np.array(item) for item in temp])
            np.save(third_path, temp)