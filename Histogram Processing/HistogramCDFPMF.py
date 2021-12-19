import matplotlib.pyplot as plt
import cv2
from collections import Counter

image = cv2.imread('/Users/rajeshr/Desktop/lc1.jpeg') #Reading Image
grayimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

row,col = grayimage.shape
print(row, col)

plt.imshow(image)
plt.title('Input grayimage')
plt.show()

flat_image = grayimage.flatten() #Converting 2D array into a single list.
plt.hist(flat_image, bins=100)
plt.title("Histogram of the Image")
plt.show()

#This function scales the y axis into 0 and 1 based on pixel counts
def softmax(arr):
    arr_sum = sum(arr)
    return [i/arr_sum for i in arr]

image_counter = Counter(flat_image)#Counts the occurence
sorted_image_counter = {k: v for k, v in sorted(image_counter.items(), key=lambda item: item[0])}#sorting the dict
softmaxed = softmax(sorted_image_counter.values())#converted the y axis

# this function calculates the cummulative distributive function
def cdf(arr):
    cdf_arr = []
    val = 0
    for i in arr:
        val += i
        cdf_arr.append(val)
    return cdf_arr

plt.plot(sorted_image_counter.keys(), cdf(softmaxed))
plt.title("CDF of the Image")
plt.show()

