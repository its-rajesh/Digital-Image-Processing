import cv2
import numpy as np
from PIL import Image as im

grayimage = cv2.imread('/Users/rajeshr/Desktop/peppers.png', cv2.IMREAD_GRAYSCALE)

grayimage = np.array(grayimage)
row, col = grayimage.shape
print(row, col)
cv2.imshow('Input Image', grayimage)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 6 bits per image
grayimageflatten = grayimage.flatten()
bit6img = ((grayimage).astype('uint8')/255)*63
# bit6img = np.array(bit6img, dtype=np.uint8)
bit6img = bit6img.reshape(row, col)

cv2.imshow('6 Bit Per Pixel', bit6img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 4 bits per image
bit4img = (grayimage/255)*15
bit4img = np.array(bit4img, dtype=np.uint8)
bit4img = bit4img.reshape(row, col)

cv2.imshow('4 Bit Per Pixel', bit4img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 2 bits per image
bit2img = [int((pixel/255)*4) for pixel in grayimageflatten]
bit2img = np.array(bit2img, dtype=np.uint8)
bit2img = bit2img.reshape(row, col)

cv2.imshow('2 Bit Per Pixel', bit2img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 1 bit per image
bit1img = [int((pixel/255)*2) for pixel in grayimageflatten]
bit1img = np.array(bit1img, dtype=np.uint8)
bit1img = bit1img.reshape(row, col)

cv2.imshow('1 Bit Per Pixel', bit1img)
cv2.waitKey(0)
cv2.destroyAllWindows()