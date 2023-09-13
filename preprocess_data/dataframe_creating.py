import os
import polars as pl
from tqdm import tqdm

class_path = "coco_classes.txt"
class_list = []
with open(class_path) as f:
    for i in f:
        class_list.append(i.strip())

cols_name = class_list + ["image_path"]

schema = [(col_name, pl.UInt8) if col_name!="image_path" else (col_name, pl.Utf8) for col_name in cols_name]
df = pl.DataFrame({}, schema=schema)

csv_path = "dataframe_Lxx.csv"
first_path = "detection_Lxx"
first_path_image = "compressed_Lxx"
second_name_list = sorted(os.listdir(first_path))
second_name_list = [item for item in second_name_list if not item.startswith('.')]
for second_name in second_name_list:
    second_path = os.path.join(first_path, second_name)
    third_name_list = sorted(os.listdir(second_path))
    third_name_list = [item for item in third_name_list if not item.startswith('.')]

    for third_name in tqdm(third_name_list):
        third_path = os.path.join(second_path, third_name)
        forth_name_list = sorted(os.listdir(third_path))
        
        for forth_name in forth_name_list:
            temp_dict = {key: 0 for key in cols_name}
            temp_dict["image_path"] = os.path.join(first_path_image, second_name, third_name, forth_name[:-4] + ".jpg")
            forth_path = os.path.join(third_path, forth_name)
            
            with open(forth_path) as f:
                for line in f:
                    idx_class = int(line.split()[0])
                    temp_dict[cols_name[idx_class]] = temp_dict.get(cols_name[idx_class], 0) + 1

            temp_df = pl.DataFrame([temp_dict], schema=schema)
            df = df.extend(temp_df)


df.write_csv(csv_path, separator=",")
print(df.estimated_size())