# import glob
# import pandas as pd

# class_path = "classes.csv"
# class_df = pd.read_csv(class_path)
# cols_name = ["image_path"] + class_df["class"].tolist()

# df = pd.DataFrame(columns=cols_name)

# first_path = "yolov8_labels"
# second_path_list = sorted(glob.glob(f"{first_path}/*"))

# for second_path in second_path_list:
#     third_path_list = sorted(glob.glob(f"{second_path}/*"))
    
#     for third_path in third_path_list:
#         temp_list = [0] * len(cols_name)
#         temp_list[0] = third_path
        
#         with open(third_path) as f:
#             for line in f:
#                 idx_class = int(line.split()[0])
#                 temp_list[idx_class] += 1
        
#         df = df.append(pd.Series(temp_list, index=cols_name), ignore_index=True)

# print(df.info())

import glob
import polars as pl
from tqdm import tqdm

class_path = "classes.csv"
class_df = pl.read_csv(class_path)
class_names = class_df.select("class")["class"].to_list()

cols_name = ["image_path"] + class_names

schema = [(col_name, pl.UInt8) if col_name!="image_path" else (col_name, pl.Utf8) for col_name in cols_name]
df = pl.DataFrame({}, schema=schema)

first_path = "yolov8_labels"
second_path_list = sorted(glob.glob(f"{first_path}/*"))

for second_path in tqdm(second_path_list):
    third_path_list = sorted(glob.glob(f"{second_path}/*"))
    
    for third_path in third_path_list:
        temp_dict = {key: 0 for key in cols_name}
        temp_dict["image_path"] = third_path
        
        with open(third_path) as f:
            for line in f:
                idx_class = int(line.split()[0])
                temp_dict[cols_name[idx_class]] = temp_dict.get(cols_name[idx_class], 0) + 1

        temp_df = pl.DataFrame([temp_dict], schema=schema)
        df = df.extend(temp_df)


csv_path = "od_dataframe.csv"
df.write_csv(csv_path, separator=",")
print(df.estimated_size())