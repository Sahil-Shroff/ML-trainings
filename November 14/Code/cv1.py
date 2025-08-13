import cv2
image=cv2.imread(r'../Data/sahil_image.jpg')
cv2.imshow('Image Window', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('output.jpg', image)