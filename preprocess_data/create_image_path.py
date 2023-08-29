import json
import glob

second_folder_path_list = glob.glob('keyframes/*')

second_folder_path_list.sort()

b = []

for i in second_folder_path_list:
    c = glob.glob(i + "/*")
    c.sort()
    b += c

for j in b[200:250]:
    print(j)


# sample_dict ={ 
#   "key1": 1, 
#   "key2": 2, 
#   "key3": 3
# }

# json_obj = json.dumps(sample_dict, indent = 3) 
# print(json_obj)

