import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.exposure import equalize_adapthist
import math


'''
______________________________________________________________________________________________
This function splits image into desired segments
______________________________________________________________________________________________
'''
def split_images(image, m, n):
  M = math.ceil(image.shape[0]/m)
  N = math.ceil(image.shape[1]/n)
  tiles = [image[x:x+M, y:y+N] for x in range(0,image.shape[0],M) for y in range(0,image.shape[1],N)]
  return tiles

'''
______________________________________________________________________________________________
This function calculated the CDF for the given image
______________________________________________________________________________________________
'''
def cdf(arr):
    cdf_arr = []
    val = 0
    for i in arr:
        val += i
        cdf_arr.append(val)

    return cdf_arr

'''
______________________________________________________________________________________________
This function calculates probability of each pixels
______________________________________________________________________________________________
'''
def softmax(arr):
    arr_sum = sum(arr)
    return [i/arr_sum for i in arr]

'''
______________________________________________________________________________________________
Performing Histogram Equalization of the individual images
______________________________________________________________________________________________
'''
def hist_spec(image):
  row, col = image.shape
  image = image.astype(float)
  unique, count = np.unique(image, return_counts=True)

  threshold = 95
  count = [min(i, threshold) for i in count]

  totpixel = sum(count)
  soft_count = softmax([i/totpixel for i in count])
  cdf_count = cdf(soft_count)

  cdf_map = dict(zip(unique, cdf_count))

  for i in range(row):
      for j in range(col):
          image[i][j] = cdf_map[image[i][j]]

  image = image*255
  x = np.array([[int(image[i,j]) for j in range(col)] for i in range(row)])
  return x

'''
______________________________________________________________________________________________
Merging segmented image back into a single image
______________________________________________________________________________________________
'''
def merge_images(images, m, n):
  hstacked = []
  for i in range(m):
    hstacked.append(np.hstack([images[i*n + j] for j in range(n)]))
  return np.vstack(hstacked)


'''
______________________________________________________________________________________________
                                        MAIN FUNCTION
______________________________________________________________________________________________
'''
# Reading gray scale image
image = cv2.imread('/Users/rajeshr/Desktop/baby.png')
grayimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
row, col = grayimage.shape

# Choose the tiling scale
m, n = 10, 10
split_image_arrays = split_images(grayimage, m, n)

# Calculating histogram spec for individual image segment
hist_speced_image = [hist_spec(img) for img in split_image_arrays]

# Merging images back
merged_image = merge_images(hist_speced_image, m, n)

# Default skimage output
'''
# THIS IS USING INBUILT SKIMAGE LIBRARY
'''
sk_output = equalize_adapthist(grayimage)
sk_output = sk_output * 255

# Displaying Input and Output Images
plt.subplot(2, 3, 1)
plt.imshow(grayimage, cmap='gray', vmin=0, vmax=255)
plt.title('Input Image')

plt.subplot(2, 3, 2)
merged_image = np.array(merged_image, dtype=np.uint8)
plt.imshow(merged_image, cmap='gray', vmin=0, vmax=255)
plt.title('With Tiling')

plt.subplot(2, 3, 3)
plt.imshow(sk_output, cmap='gray', vmin=0, vmax=255)
plt.title('SkImage CLAHE output')

plt.suptitle("Contrast Limited Adaptive Histogram Equalization")
plt.tight_layout()
plt.show()
