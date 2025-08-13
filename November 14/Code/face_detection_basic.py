import cv2

# Load the cascade classifier
face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

# Read the image
image= cv2.imread('../Data/sahil_image.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect faces
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

cv2.imshow('Face Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()