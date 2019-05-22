import cv2
import numpy as np

img = cv2.imread('./P0000_0768_0768.png')

xss = [[128, 135, 130, 123], [180, 187, 181, 174], 
       [141, 153, 159, 146], [168, 175, 171, 164],
       [301, 293, 300, 307], [330, 337, 345, 337],
       [330, 337, 334, 325], [658, 648, 657, 667]]

yss = [[253, 256, 270, 267], [300, 301, 318, 316], 
       [285, 272, 279, 291], [295, 297, 312, 309], 
       [741, 736, 721, 724], [752, 749, 766, 768],
       [714, 716, 735, 733], [242, 237, 222, 225]]

colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]

for xs, ys in zip(xss, yss):
    # corners
    for x, y, color in zip(xs, ys, colors):
        cv2.circle(img, (np.int0(x), np.int0(y)), 3, color, -1)

    # edges
    pts = np.array([[x, y] for x, y in zip(xs, ys)])
    pts = pts.reshape((-1,1,2))
    cv2.polylines(img, [pts], True, (0,255,255))

cv2.imshow('img', img)
cv2.waitKey()

