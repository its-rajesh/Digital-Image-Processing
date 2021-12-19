import cv2

img = cv2.imread('/Users/rajeshr/Desktop/view4.png')
grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow('Original Image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('Gray Image',grayImage)
cv2.waitKey(0)
cv2.destroyAllWindows()