import json
import glob

first_path = "compressed_Lxx"
json_file_path = "full_path.json"

all_path_list = []
second_path_list = sorted(glob.glob(f"{first_path}/*"))
for second_path in second_path_list:
    third_path_list = sorted(glob.glob(f"{second_path}/*"))
    for third_path in third_path_list:
        forth_path_list = sorted(glob.glob(f"{third_path}/*"))
        all_path_list += forth_path_list

idx_list = list(range(len(all_path_list)))
path_dict = dict(zip(idx_list, all_path_list))

with open(json_file_path, "w") as json_file:
    json.dump(path_dict, json_file)