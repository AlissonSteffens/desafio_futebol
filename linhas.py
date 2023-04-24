import cv2
import numpy as np
import sys

img_name = sys.argv[1]
frame = cv2.imread(img_name)
hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
mask_green = cv2.inRange(hsv, (36, 25, 25), (86, 255, 255)) # green mask to select only the field
frame_masked = cv2.bitwise_and(frame, frame, mask=mask_green)

gray = cv2.cvtColor(frame_masked, cv2.COLOR_RGB2GRAY)
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

canny = cv2.Canny(gray, 50, 150, apertureSize=3)
# Hough line detection
lines = cv2.HoughLinesP(canny, 1, np.pi / 180, 50, None, 50, 20)

# zerar a imagem transformada
image_lines = np.zeros(frame.shape, np.uint8)

# atualizar a imagem transformada desenhando as linhas encontradas
if lines is not None:
    for i in range(0, len(lines)):
        l = lines[i][0]
        # garantee points are integers
        l = [int(x) for x in l]
        cv2.line(image_lines, (l[0], l[1]), (l[2], l[3]), (255, 255, 255), 3, cv2.LINE_AA)




cv2.imshow("frame", frame)
cv2.imshow("canny", canny)
cv2.imshow("lines", image_lines)
cv2.waitKey(0)
cv2.destroyAllWindows()