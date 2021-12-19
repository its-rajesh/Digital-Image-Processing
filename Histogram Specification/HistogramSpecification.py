import numpy as np
import cv2
import matplotlib.pyplot as plt

'''
______________________________________________________________________________________________
This function calculates the Cummulative distributive function of the given softmaxed function
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
____________________________________________________________________________
This function calculates the probability of each pixel
____________________________________________________________________________
'''
def softmax(arr):
    arr_sum = sum(arr)
    return [i/arr_sum for i in arr]

'''
______________________________________________________________________________________________
This function takes image input and returns its CDF and Histogram equalized image
______________________________________________________________________________________________
'''

def hist_spec(image):
  row, col = image.shape
  image = image.astype(float)
  unique, count = np.unique(image, return_counts=True)
  totpixel = sum(count)
  soft_count = softmax([i/totpixel for i in count])
  cdf_count = cdf(soft_count)

  cdf_map = dict(zip(unique, cdf_count))

  for i in range(row):
      for j in range(col):
          image[i][j] = cdf_map[image[i][j]]

  image = image*255
  x = np.array([[int(image[i,j]) for j in range(col)] for i in range(row)])
  return cdf_map, x

'''
______________________________________________________________________________________________
                                    MAIN FUNCTION
______________________________________________________________________________________________
'''

# Reading image inputs: Input Image and Reference Image
reference_image = cv2.imread('/Users/rajeshr/Desktop/tulips.jpeg')
input_image = cv2.imread('/Users/rajeshr/Desktop/lc1.jpeg')

# Converting them into gray scale
reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

# Calculating its CDF and Histogram Equalized Image
reference_cdf_map, reference_cdf_image = hist_spec(reference_image)
input_cdf_map, input_cdf_image = hist_spec(input_image)

# Displaying the Reference Image Data
f = plt.figure()
f.set_figwidth(14)
f.set_figheight(7)

plt.subplot(3, 5, 6)
plt.imshow(reference_image, cmap='gray', vmin=0, vmax=255)
plt.title("Reference Image")

plt.subplot(3, 5, 7)
plt.hist(reference_image.flatten(), bins=100)
plt.title('Histogram')

plt.subplot(3, 5, 8)
plt.plot(list(reference_cdf_map.keys()), list(reference_cdf_map.values()))
plt.title('CDF')

plt.subplot(3, 5, 9)
reference_cdf_image = np.array(reference_cdf_image, dtype=np.uint8)
plt.imshow(reference_cdf_image, cmap='gray', vmin=0, vmax=255)
plt.title('Histogram Equalized')

plt.subplot(3, 5, 10)
plt.hist(reference_cdf_image.flatten(), bins=100)
plt.title('Output Histogram')

# Displaying the Input Image Data
plt.subplot(3, 5, 1)
plt.imshow(input_image, cmap='gray', vmin=0, vmax=255)
plt.title('Input Image')

plt.subplot(3, 5, 2)
plt.hist(input_image.flatten(), bins=100)
plt.title('Histogram')

plt.subplot(3, 5, 3)
plt.plot(list(input_cdf_map.keys()), list(input_cdf_map.values()))
plt.title('CDF')

plt.subplot(3, 5, 4)
input_cdf_image = np.array(input_cdf_image, dtype=np.uint8)
plt.imshow(input_cdf_image, cmap='gray', vmin=0, vmax=255)
plt.title('Histogram Equalized')

plt.subplot(3, 5, 5)
plt.hist(input_cdf_image.flatten(), bins=100)
plt.title('Histogram Output')

'''
______________________________________________________________________________________________
Perfroming Histogram Specification or Histogram Matching
______________________________________________________________________________________________
'''

matched_image = np.zeros(input_cdf_image.shape)
row, col = matched_image.shape

reference_cdf_unique_values = np.unique(reference_cdf_image)

for i in range(row):
  for j in range(col):
    for index in range(input_cdf_image[i][j]):
      if (input_cdf_image[i][j]-index) in reference_cdf_unique_values:
        matched_image[i][j] = (input_cdf_image[i][j]-index)
        break

# Displaying the Histogram result
matched_image = np.array(matched_image, dtype=np.uint8)
plt.subplot(3, 5, 13)
plt.imshow(matched_image, cmap='gray', vmin=0, vmax=255)
plt.title('Histogram Matched Image')

#plt.subplot(3, 5, 12)
#plt.hist(matched_image.flatten(), bins=100)
#plt.title('Matched Histogram')

plt.suptitle("Histogram Specification of Gray Image")

plt.tight_layout()
plt.show()