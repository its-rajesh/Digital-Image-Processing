import cv2
import numpy as np

grayimage = cv2.imread('/Users/rajeshr/Desktop/lc1.jpeg', cv2.IMREAD_GRAYSCALE)

row, col = grayimage.shape
#print(row, col)

flat_image = grayimage.flatten() #Converting 2D array into a single list.
#print(flat_image, len(flat_image))
Max, Min = max(flat_image), min(flat_image)
#print(Max, Min)

stretched = [((pixel-Min)/(Max-Min))*255 for pixel in flat_image]
stretched = np.array(stretched, dtype=np.uint8)

'''
out = np.array_split(stretched, row)
out = np.array(out)
print(out.shape)
print(out)
'''

out = stretched.reshape(row, col)
#print(out.shape)


cv2.imshow('Input Image', grayimage)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('Output Image', out)
cv2.waitKey(0)
cv2.destroyAllWindows()
