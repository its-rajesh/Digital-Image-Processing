import cv2
import numpy as np
from collections import Counter
from matplotlib import pyplot as plt

def cdf(arr):
    cdf_arr = []
    val = 0
    for i in arr:
        val += i
        cdf_arr.append(val)

    return cdf_arr

def softmax(arr):
    arr_sum = sum(arr)
    return [i/arr_sum for i in arr]

image = cv2.imread('/Users/rajeshr/Desktop/lc1.jpeg')
grayimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

print(grayimage.shape)
row, col = grayimage.shape

newimage = grayimage.copy()
newimage = newimage.astype(float)

unique, count = np.unique(grayimage, return_counts=True)
plt.bar(unique, count)
plt.title("Historgram of Input")
plt.xlabel("Pixel Intensitiy")
plt.ylabel("Occurence")
plt.show()

totpixel = sum(count)

soft_count = softmax([i/totpixel for i in count])
cdf_count = cdf(soft_count)
plt.plot(unique, cdf_count)
plt.title("CDF of Input")
plt.xlabel("Pixel Intensitiy")
plt.ylabel("Probability")
plt.show()

cdf_map = dict(zip(unique, cdf_count))

for i in range(row):
    for j in range(col):
        newimage[i][j] = cdf_map[grayimage[i][j]]

newimage = newimage * 255

x = np.array([[int(newimage[i,j]) for j in range(row)] for i in range(col)], dtype=np.uint8)

cv2.imshow("Histogram Equilized",x)
cv2.waitKey(0)
cv2.destroyAllWindows()