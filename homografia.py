import cv2
import numpy as np

# Carregar as imagens
img_transformada = cv2.imread('Teste_1.jpg')
img_campo = cv2.imread('campo.jpeg')

# manter apenas linhas retas e arcos na imagem original
img_campo = cv2.cvtColor(img_campo, cv2.COLOR_BGR2GRAY)
img_campo = cv2.Canny(img_campo, 100, 200)

# Fechamento morfológico para preencher os as linhas duplas
kernel = np.ones((5, 5), np.uint8)
img_campo = cv2.morphologyEx(img_campo, cv2.MORPH_CLOSE, kernel)


#  manter apenas linhas retas e arcos na imagem transformada
hsv = cv2.cvtColor(img_transformada, cv2.COLOR_RGB2HSV)
mask_green = cv2.inRange(hsv, (36, 25, 25), (86, 255, 255)) # green mask to select only the field
frame_masked = cv2.bitwise_and(img_transformada, img_transformada, mask=mask_green)

gray = cv2.cvtColor(frame_masked, cv2.COLOR_RGB2GRAY)
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

canny = cv2.Canny(gray, 50, 150, apertureSize=3)
# Hough line detection
lines = cv2.HoughLinesP(canny, 1, np.pi / 180, 50, None, 50, 20)

# zerar a imagem transformada
img_transformada = np.zeros(img_transformada.shape, np.uint8)

# atualizar a imagem transformada desenhando as linhas encontradas
if lines is not None:
    for i in range(0, len(lines)):
        l = lines[i][0]
        # garantee points are integers
        l = [int(x) for x in l]
        cv2.line(img_transformada, (l[0], l[1]), (l[2], l[3]), (255, 255, 255), 3, cv2.LINE_AA)


# limpando a imagem transformada para remover ruídos com abertura e fechamento morfológicos
kernel = np.ones((5, 5), np.uint8)
img_transformada = cv2.morphologyEx(img_transformada, cv2.MORPH_OPEN, kernel)
img_transformada = cv2.morphologyEx(img_transformada, cv2.MORPH_CLOSE, kernel)



# ---------------

# Inicializar o detector e extrator de recursos
detector = cv2.xfeatures2d.SIFT_create()
extrator = cv2.xfeatures2d.SIFT_create()

# Encontrar os pontos de interesse e os descritores
kp1, descritores1 = detector.detectAndCompute(img_transformada, None)
kp2, descritores2 = extrator.detectAndCompute(img_campo, None)

# Criar o objeto FLANN
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Encontrar os melhores matches
matches = flann.knnMatch(descritores2, descritores1, k=2)

# Armazenar os melhores matches usando o teste de Lowe
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)

# Encontrar os pontos de interesse correspondentes
query_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
train_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)


# desenhar os pontos
img3 = cv2.drawMatches(img_transformada, kp1, img_campo, kp2, good, None, flags=2)
cv2.imshow('Imagem Original', img3)


# # Encontrar a homografia
# matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)

# # Aplicar a homografia
# matches_mask = mask.ravel().tolist()
# h, w = img_campo.shape
# pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
# dst = cv2.perspectiveTransform(pts, matrix)
# img_homografada = cv2.polylines(img_transformada, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)



# Exibir as imagens
# cv2.imshow('Imagem Original', img_campo)
# cv2.imshow('Imagem Transformada', img_transformada)
# cv2.imshow('Imagem Homografada', img_homografada)
cv2.waitKey(0)
cv2.destroyAllWindows()

