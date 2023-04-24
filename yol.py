import torch
import cv2
import numpy as np
import sys

def plot_one_box(points, img, color=None, label=None, line_thickness=None):
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(points[0]), int(points[1])), (int(points[2]), int(points[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def get_bounding_box_main_color(img, bounding_box):
    x1, y1, x2, y2 = bounding_box
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    cropped = img[y1:y2, x1:x2]
    # remove greens from image (field)
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(hsv, (36, 25, 25), (86, 255, 255))  # green mask to select only the field
    mask_green = cv2.bitwise_not(mask_green)
    cropped = cv2.bitwise_and(cropped, cropped, mask=mask_green)
    # get most common color
    colors, count = np.unique(cropped.reshape(-1, cropped.shape[-1]), axis=0, return_counts=True)
    color = colors[count.argmax()]

    return color.tolist()

def get_bounding_box_center(bounding_box):
    x1, y1, x2, y2 = bounding_box
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def get_bounding_box_player_foot(bounding_box):
    x1, y1, x2, y2 = bounding_box
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    return int((x1 + x2) / 2), int(y2)

# Model
model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # or yolov5n - yolov5x6, custom

# Image from argument
img = sys.argv[1]
# Inference
results = model(img)

# load image with OpenCV, to draw on it later
img = cv2.imread(img)

# get results keys
print(results)

# get bounding boxes
boundingBoxes = results.xyxy[0]



# Results
boundingBoxes = results.xyxy[0]

# Draw bounding boxes and labels of detections
for *xyxy, conf, cls in reversed(boundingBoxes):
    # only continue if is a person
    if model.names[int(cls)] != "person":
        continue
    label = f"{model.names[int(cls)]} {conf:.2f}"
    plot_one_box(xyxy, img, color=get_bounding_box_main_color(img, xyxy), label=label, line_thickness=3)


# Draw player center over bounding box
for *xyxy, conf, cls in reversed(boundingBoxes):
    # only continue if is a person
    if model.names[int(cls)] != "person":
        continue
    center = get_bounding_box_player_foot(xyxy)
    # draw point
    cv2.circle(img, center, 5, (0, 0, 255), 2)
    
# Show image
cv2.imshow("image", img)

# wait until any key is pressed
cv2.waitKey(0)