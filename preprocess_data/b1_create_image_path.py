import json
import glob

def write_json(result, json_file_path):
    with open(json_file_path, "w") as json_file:
        json.dump(result, json_file)

if __name__ == "__main__":
    first_path = "compressed_Lxx"
    json_file_paths = ["keydata/full_path_v1.json", "keydata/full_path_v3.json", "keydata/full_path_v5.json"]

    all_path_list = []
    second_path_list = sorted(glob.glob(f"{first_path}/*"))
    for second_path in second_path_list:
        third_path_list = sorted(glob.glob(f"{second_path}/*"))
        for third_path in third_path_list:
            forth_path_list = sorted(glob.glob(f"{third_path}/*"))
            all_path_list += forth_path_list

    idx_list = list(range(len(all_path_list)))
    path_dict = dict(zip(idx_list, all_path_list))
    write_json(path_dict, json_file_paths[0])

    # for vectors v3
    result_v3 = {key: value for key, value in path_dict.items() if key % 3 != 1}
    result_v3 = {key: value for key, value in enumerate(result_v3.values())}
    write_json(result_v3, json_file_paths[1])

    # for vectors v5
    result_v5 = {key: value for key, value in path_dict.items() if key % 3 == 1}
    result_v5 = {key: value for key, value in enumerate(result_v5.values())}
    write_json(result_v5, json_file_paths[2])