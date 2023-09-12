import numpy as np
import polars as pl
from functools import reduce

# 1. Sau khi dùng text-search thì mình thu được 200 phần tử được chứa trong clip_filter, hình dạng như nào thì anh print ra xem nhé
clip_filter_path = "abc.npy"
clip_filter = list(np.load(clip_filter_path, allow_pickle=True))

print(clip_filter[:2])
print(type(clip_filter))


# 2. Đây là yêu cầu lọc của phần Object Detection, ảnh phải chứa ít nhất 3 object person
detection_query = {'person': 3}
# object_filter = {'person': 3, 'car': 2}


# 3. Còn đây là toàn bộ dataframe của mình:
dataframe_path = "dataframe_L03.csv" # Ở DF này, 1 hàng sẽ bao gồm image_path (ở cột cuối), và số lượng phần tử của mỗi class
pldf = pl.read_csv(dataframe_path) # pddf là Polars DataFrame đó anh
temp_pl = pldf.filter(pldf["person"] >= 3).head() # Em thử lọc đơn giản, nhưng mình sẽ lọc phức tạp hơn
print(temp_pl.shape)

# clip_filter_pl = pl.DataFrame(clip_filter).rename({"imgpath": "image_path"})
# print(clip_filter_pl.head())

# ls = [x['imgpath'] for x in clip_filter]
# polar_df = pldf.join(clip_filter_pl, on='image_path')
# print(polar_df.head())
# print(polar_df.shape)



'''
Q: Giờ mình cần rerank lại cái clip_filter kia. 
Bằng cách: Tất cả những ảnh trong clip_filter mà thỏa mãn detection_query thì anh để lên đầu. 
            VD: detection_query = {'person': 3, 'car': 2}, thì ảnh nào có person >= 3 và car >= 2 thì được tính là thỏa mãn.
            (Nếu xếp phần thỏa mãn theo thứ tự tăng dần của tổng các phần tử thì tốt ạ).
            Còn lại để xuống dưới.
            
Định dạng và số lượng phần tử: Vẫn phải giống clip_filter anh nhé.
'''
def filter_df(df, conditions):
    if not conditions:
        # If no conditions provided, return the original DataFrame
        return df
    else:
        # Combine conditions using logical AND
        combined_condition = conditions[0]
        for condition in conditions[1:]:
            combined_condition = combined_condition & condition

        # Filter the DataFrame
        filtered_df = df.filter(combined_condition)
        return filtered_df
    

def and_operation(bool_list):
    return reduce((lambda x, y: x & y), bool_list)

def not_and_operation(bool_list):
    return not reduce((lambda x, y:  not x & y), bool_list)

def gen_np_from_df(polar_df):
    df = polar_df.select(['image_path', 'id'])
    filters = [{'imgpath': x[0], 'id': x[1]} for x in list(df.iter_rows())]
    return np.array(filters)

def rerank(clip_filter, detection_query):
    # init 
    clip_filter_pl = pl.DataFrame(clip_filter).rename({"imgpath": "image_path"})
    polar_df = pldf.join(clip_filter_pl, on='image_path')

    # filter
    conditions = [polar_df[x] >= detection_query[x]  for x in detection_query.keys()]
    not_conditions = [polar_df[x] < detection_query[x]  for x in detection_query.keys()]
    
    filter_pl = filter_df(polar_df, conditions)
    filter_pl = filter_pl.sort(filter_pl.columns)
    not_filter_pl = filter_df(polar_df, not_conditions)
    filter_result = filter_pl.vstack(not_filter_pl)
    # filter_result = filter_result.sort(filter_result.columns)
    print(filter_result.head())
    filter_np = gen_np_from_df(filter_result)
    return filter_np

new_clip_features = rerank(clip_filter, detection_query)
print(new_clip_features[:5], len(new_clip_features))