import cv2
image=cv2.imread(r'../Data/sahil_image.jpg')

cv2.imshow('Image Window', image)

resized_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY, (64, 64))
cv2.imshow('Image Window', resized_image)
cv2.waitKey(0)

hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.imshow('Image Window', hsv_image)
cv2.waitKey(0)

flipped_image = cv2.flip(image, 1)
cv2.imshow('Image Window', flipped_image)
cv2.waitKey(0)

cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('cv2_output.jpg', image)