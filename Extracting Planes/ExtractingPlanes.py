import cv2
import numpy as np

img = cv2.imread('/Users/rajeshr/Desktop/view4.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
row, col, plane = img.shape
#r, g, b = cv2.split(img)


temp = np.zeros((row,col,plane),np.uint8)
temp[:,:,0] = img[:,:,0] #Blue Plane
cv2.imshow('Blue Plane',temp)
cv2.waitKey(0)
cv2.destroyAllWindows()

temp = np.zeros((row,col,plane),np.uint8)
temp[:,:,1] = img[:,:,1] #Green Plane
cv2.imshow('Green Plane',temp)
cv2.waitKey(0)
cv2.destroyAllWindows()

temp = np.zeros((row,col,plane),np.uint8)
temp[:,:,2] = img[:,:,2] #Red Plane
cv2.imshow('Red Plane',temp)
cv2.waitKey(0)
cv2.destroyAllWindows()


'''
img = cv.imread('/Users/rajeshr/Desktop/view4.png')
b, g, r = cv.split(img)

fig, ax = plt.subplots(1, 3, figsize=(16, 8))
fig.tight_layout()

ax[0].imshow(r)
ax[0].set_title("Red")

ax[1].imshow(g)
ax[1].set_title("Green")

ax[2].imshow(b)
ax[2].set_title("Green")

#ax[2].imshow(cv.cvtColor(b, cv.COLOR_BGR2RGB))
#ax[2].set_title("Blue")

plt.show()
'''