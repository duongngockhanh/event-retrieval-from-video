import numpy as np
import polars as pl

# 1. Sau khi dùng text-search thì mình thu được 200 phần tử được chứa trong clip_filter, hình dạng như nào thì anh print ra xem nhé
clip_filter_path = "abc.npy"
clip_filter = list(np.load(clip_filter_path, allow_pickle=True))

print(type(clip_filter))


# 2. Đây là yêu cầu lọc của phần Object Detection, ảnh phải chứa ít nhất 3 object person
detection_query = {'person': 3}
# object_filter = {'person': 3, 'car': 2}


# 3. Còn đây là toàn bộ dataframe của mình:
dataframe_path = "dataframe_L03.csv" # Ở DF này, 1 hàng sẽ bao gồm image_path (ở cột cuối), và số lượng phần tử của mỗi class
pldf = pl.read_csv(dataframe_path) # pddf là Polars DataFrame đó anh
temp_pl = pldf.filter(pldf["person"] >= 3).head() # Em thử lọc đơn giản, nhưng mình sẽ lọc phức tạp hơn


'''
Q: Giờ mình cần rerank lại cái clip_filter kia. 
Bằng cách: Tất cả những ảnh trong clip_filter mà thỏa mãn detection_query thì anh để lên đầu. 
            VD: detection_query = {'person': 3, 'car': 2}, thì ảnh nào có person >= 3 và car >= 2 thì được tính là thỏa mãn.
            (Nếu xếp phần thỏa mãn theo thứ tự tăng dần của tổng các phần tử thì tốt ạ).
            Còn lại để xuống dưới.
            
Định dạng và số lượng phần tử: Vẫn phải giống clip_filter anh nhé.
'''