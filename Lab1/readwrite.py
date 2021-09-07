import cv2

image = cv2.imread("/Users/rajeshr/Desktop/view4.png")
print(image)
cv2.imshow('Input Image',image)
cv2.waitKey(0)
cv2.destroyAllWindow