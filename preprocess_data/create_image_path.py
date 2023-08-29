import json
import glob

second_folder_path_list = glob.glob('keyframes/*')

second_folder_path_list.sort()

all_path_list = []

for i in second_folder_path_list:
    third_path = glob.glob(i + "/*")
    third_path.sort()
    all_path_list += third_path

idx = list(range(len(all_path_list)))

index_path_dict = dict(zip(idx, all_path_list))


json_file_path = "index_path.json"

# Mở tệp JSON để ghi dữ liệu
with open(json_file_path, "w") as json_file:
    json.dump(index_path_dict, json_file)

