import cv2
from matplotlib import pyplot as plt
import numpy as np

img = cv2.imread('/Users/rajeshr/Desktop/7.jpeg')
#cv2.imshow('Input Image', img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

size = 15
motion_blur = np.zeros((size, size))
motion_blur[int((size-1)/2), :] = np.ones(size)
motion_blur = motion_blur / size

# applying the kernel to the input image
output = cv2.filter2D(img, -1, motion_blur)
#cv2.imshow('Motion Blur', output)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(output),plt.title('Motion Blurred')
plt.xticks([]), plt.yticks([])
plt.show()