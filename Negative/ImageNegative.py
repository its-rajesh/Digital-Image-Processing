import cv2

img = cv2.imread('/Users/rajeshr/Desktop/peppers.png')
negativefruitimg = 255-img

cv2.imshow('Original Image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('Negative Image',negativefruitimg)
cv2.waitKey(0)
cv2.destroyAllWindows()

