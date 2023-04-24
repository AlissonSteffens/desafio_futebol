import cv2
import numpy as np
import sys

# read positions.txt as space separated values indicating index, x and y
positions = np.loadtxt("positions.txt", dtype=int)

# read homography.txt as a 3x3 matrix
M = np.loadtxt("homography.txt")

# read image
img = cv2.imread("campo.jpeg")

def get_random_color():
    return (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))

def is_outside_of_the_field(x, y):
  margin = 100
  # consider the field as a rectangle and with a margin of 10 pixels
  return x < margin or x > img.shape[1] - margin or y < margin or y > img.shape[0] - margin

valid_players = {}

# draw points
for index, x, y in positions:
    # get point
    point = np.array([x, y, 1])
    # apply homography
    point = M.dot(point)
    # normalize
    point = point / point[2]
    # check if the point is outside of the field
    if is_outside_of_the_field(point[0], point[1]):
      cv2.circle(img, (int(point[0]), int(point[1])), 1, (0, 0, 255), 2)
      continue
    # draw point
    cv2.circle(img, (int(point[0]), int(point[1])), 1, (255, 0, 0), 2)
    # draw image from players folder
    player_img = cv2.imread(f"players/{index}.jpg")
    # resize image
    player_img = cv2.resize(player_img, (20, 40))
    # get the bottom center of the image
    bottom_center = (int(point[0] - player_img.shape[1] / 2), int(point[1] - player_img.shape[0]))
    # add image to the field
    img[bottom_center[1]:bottom_center[1] + player_img.shape[0], bottom_center[0]:bottom_center[0] + player_img.shape[1]] = player_img



    # draw number with random color and white background
    cv2.putText(img, str(index), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 4, cv2.LINE_AA)
    cv2.putText(img, str(index), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, .5, get_random_color(), 1, cv2.LINE_AA)
    # add to valid players
    valid_players[index] = (int(point[0]), int(point[1]))

# save valid players to a file
with open("valid_players.txt", "w") as f:
  for index, (x, y) in valid_players.items():
    f.write(f"{index} {x} {y}\n")


    

# show image
cv2.imshow("image", img)
cv2.waitKey(0)
