import sys
import cv2
import numpy as np

# load image with OpenCV, to draw on it later
img = cv2.imread(sys.argv[1])
# load campo.jpeg for reference
campo = cv2.imread("campo.jpeg")

# draw up to 4 points on the image
points = []
def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(img, (x, y), 5, (0, 0, 255), 2)
        # draw number
        cv2.putText(img, str(len(points)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("image", img)

cv2.namedWindow("image")
cv2.setMouseCallback("image", draw_circle)

while True:
    cv2.imshow("image", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if len(points) == 4:
        break


# get the 4 points from the image
pts1 = np.float32(points)

# let the user draw the 4 points on the reference image
points = []
def draw_circle_other(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(campo, (x, y), 5, (0, 0, 255), 2)
        cv2.imshow("campo", campo)

cv2.namedWindow("campo")
cv2.setMouseCallback("campo", draw_circle_other)

while True:
    cv2.imshow("campo", campo)
    cv2.imshow("image", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if len(points) == 4:
        break

# get the 4 points from the reference image
pts2 = np.float32(points)

print(pts1)
print(pts2)

# get the homography
M = cv2.getPerspectiveTransform(pts1, pts2)


# reload image
img = cv2.imread(sys.argv[1])
# apply the homography to the image
dst = cv2.warpPerspective(img, M, (campo.shape[1], campo.shape[0]))

# field in the background
dst = cv2.addWeighted(dst, 0.9, campo, 0.1, 0)

# show the result
cv2.imshow("result", dst)
cv2.waitKey(0)