import cv2
import numpy as np

image = cv2.imread("/Users/rajeshr/Desktop/lc1.jpeg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Input Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

row, col = image.shape
flat_image = image.flatten()
new_image = []

for pixel in flat_image:
    if pixel <= 100:
        new_image.append(255)
    else:
        new_image.append(0)

new_image = np.array(new_image)
new_image = new_image.reshape(row,col)
new_image = np.array(new_image, dtype=np.uint8)
cv2.imshow('Output Image', new_image)
cv2.waitKey(0)
cv2.destroyAllWindows()