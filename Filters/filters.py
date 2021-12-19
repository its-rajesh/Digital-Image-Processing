import cv2
from matplotlib import pyplot as plt

img = cv2.imread('/Users/rajeshr/Desktop/33.jpeg')

size = [3, 11, 21]
for i in range(3):
    blur = cv2.blur(img,(size[i],size[i]))

    plt.subplot(121),plt.imshow(img),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
    plt.xticks([]), plt.yticks([])
    plt.title("Averaging filter with size "+str(size[i]))
    plt.show()

size = [5, 21]
sigma = [1, 2, 5]
for i in range(3):
    for j in range(2):
        blur = cv2.GaussianBlur(img,(size[j],size[j]),sigma[i])
        plt.subplot(121),plt.imshow(img),plt.title('Original')
        plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
        plt.xticks([]), plt.yticks([])
        plt.title("Gaussian filter with size "+str(size[j])+" and sigma "+str(sigma[i]))
        plt.show()