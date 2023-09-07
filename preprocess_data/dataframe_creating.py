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

first_name = "detection_L03"
second_name_list = sorted(os.listdir(first_name))

first_name_image = "L03_keyframes"

for second_name in tqdm(second_name_list):
    third_name_list = sorted(os.listdir(os.path.join(first_name, second_name)))
    
    for third_name in third_name_list:
        temp_dict = {key: 0 for key in cols_name}
        temp_dict["image_path"] = os.path.join(first_name_image, second_name, third_name[:-4] + ".jpg")
        third_path = os.path.join(first_name, second_name, third_name)
        
        with open(third_path) as f:
            for line in f:
                idx_class = int(line.split()[0])
                temp_dict[cols_name[idx_class]] = temp_dict.get(cols_name[idx_class], 0) + 1

        temp_df = pl.DataFrame([temp_dict], schema=schema)
        df = df.extend(temp_df)


csv_path = "dataframe_L03.csv"
df.write_csv(csv_path, separator=",")
print(df.estimated_size())