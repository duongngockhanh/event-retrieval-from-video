import cv2
import numpy as np
import json
import pandas as pd

def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union valueq
	return iou

if __name__ == "__main__":

    path = "0011.json"
    img_path = "0011.jpg"
    
    with open(path) as f:
        data = json.load(f)

    box_list = []
    keys = list(data.keys()) # 'detection_class_names', 'detection_class_labels', 'detection_scores', 'detection_boxes', 'detection_class_entities'
    number_boxes = len(data[keys[0]])
    threshold_nms = 0.15

    shape_image = (720, 1280)
    
    for i in range(number_boxes):
        temp_float = list(map(float, [data[keys[3]][i][1], data[keys[3]][i][0], data[keys[3]][i][3], data[keys[3]][i][2]]))
        temp_pixel = list(map(int, [temp_float[0] * shape_image[1], temp_float[1] * shape_image[0], temp_float[2] * shape_image[1], temp_float[3] * shape_image[0]]))
        temp_element = [int(data[keys[1]][i]), float(data[keys[2]][i]), temp_pixel, data[keys[4]][i]]
        box_list.append(temp_element)
    
    final_boxes = []
    while len(box_list) > 0:
        a = box_list.pop(0)
        final_boxes.append(a)
        i = 0
        while i < len(box_list):
            if bb_intersection_over_union(box_list[i][2], a[2]) > threshold_nms:
                b = box_list.pop(i)
                i -= 1
            i += 1
    
    img = cv2.imread(img_path)
    
    for i in final_boxes:
        img = cv2.rectangle(img, i[2][0: 2], i[2][2: 4], color=(0, 255, 0), thickness=2)
        img = cv2.putText(img, i[3], i[2][0: 2], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
    
    cv2.imwrite("after_nms.jpg", img)