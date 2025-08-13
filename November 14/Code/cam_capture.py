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

    cv2.imshow('Webcam Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()