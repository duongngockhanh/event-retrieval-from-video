import numpy as np

path = "clip_features_L03/L03_V001.npy"

a = np.load(path)

print(a.shape) # (1173, 1, 512)