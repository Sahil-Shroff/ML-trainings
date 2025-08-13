import cv2

# Video capture device
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print('Error: Could not open webcam.')
    exit()

while True:
    # capture frame by frame
    ret, frame = cap.read()
    if not ret:
        print('Error: Failed to grab frame.')
        break

    # Load the cascade classifier
    face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

    # Read the image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, 'top-left: ' + str(x) + ', ' + str(y), (x-20, y), font, .4, (0, 0, 255), 3, cv2.LINE_4)
        cv2.putText(frame, 'top-right: ' + str(x+w) + ', ' + str(y), (x+w-20, y), font, .4, (0, 0, 255), 3, cv2.LINE_4)
        cv2.putText(frame, 'bottom-left: ' + str(x) + ', ' + str(y+h), (x-20, y+h), font, .4, (0, 0, 255), 3, cv2.LINE_4)
        cv2.putText(frame, 'bottom-right: ' + str(x+w) + ', ' + str(y+h), (x+w-20, y+h), font, .4, (0, 0, 255), 3, cv2.LINE_4)

    cv2.imshow('Webcam Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()