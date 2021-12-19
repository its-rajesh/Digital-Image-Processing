import cv2
import numpy as np

grayimage = cv2.imread('/Users/rajeshr/Desktop/lc1.jpeg', cv2.IMREAD_GRAYSCALE)
row, col = grayimage.shape

flat_image = grayimage.flatten() #Converting 2D array into a single list.
Max, Min = max(flat_image), min(flat_image)

stretched = [((pixel-Min)/(Max-Min))*255 for pixel in flat_image]
stretched = np.array(stretched, dtype=np.uint8)
out = stretched.reshape(row, col)

cv2.imshow('Input Image', grayimage)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('Output Image', out)
cv2.waitKey(0)
cv2.destroyAllWindows()
