import cv2
import json

image_path = "0011.jpg"
json_path = "0011.json"

image = cv2.imread(image_path)

with open(json_path) as f:
    data = json.load(f)

keys = list(data.keys())

bounding_boxes = data[keys[3]]

classes = data[keys[4]]


for i in range(len(classes)):
    # if classes[i] == "Man" or classes[i] == "Woman" or classes[i] == "Person":
    x_min = int(float(bounding_boxes[i][1]) * image.shape[1])
    y_min = int(float(bounding_boxes[i][0]) * image.shape[0])
    x_max = int(float(bounding_boxes[i][3]) * image.shape[1])
    y_max = int(float(bounding_boxes[i][2]) * image.shape[0])

    color = (0, 255, 0)
    thickness = 2
    
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)

cv2.imwrite("before_nms.jpg", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
