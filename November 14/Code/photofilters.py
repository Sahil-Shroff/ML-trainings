import cv2
import numpy as np

def apply_grayscale(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def apply_sepia(frame):
    img_sepia = np.array(frame, dtype=np.float64)
    img_sepia = cv2.transform(img_sepia, np.matrix([[.272, .534, .131],
                                                    [.349, .686, .168],
                                                    [.393, .769, .189]]))
    
    img_sepia[np.where(img_sepia > 255)] = 255
    return np.array(img_sepia, dtype=np.uint8)

def apply_negative(frame):
    return cv2.bitwise_not(frame)

def apply_blur(frame):
    return cv2.GaussianBlur(frame, (15, 15), 0)

def apply_edge_detection(frame):
    return cv2.Canny(frame, 100, 200)

filters = {
    'g': apply_grayscale,
    's': apply_sepia,
    'n': apply_negative,
    'b': apply_blur,
    'e': apply_edge_detection,
    'o': lambda x: x # Original
}

# Video capture device
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print('Error: Could not open webcam.')
    exit()

current_filter = 'o' # Start with Original

while True:
    # capture frame by frame
    ret, frame = cap.read()
    if not ret:
        print('Error: Failed to grab frame.')
        break

    # Apply the current filter
    filtered_frame = filters[current_filter](frame)

    font = cv2.FONT_ITALIC
    instructions = "Press 'g': Grayscale | 's': Sepia | 'n': Negative | 'b': Blur | 'e': Edge | 'o': Original | 'q': Quit"
    y0, dy = 20, 40
    for i, line in enumerate(instructions.split('|')):
        y = y0 + i*dy
        cv2.putText(filtered_frame, line.strip(), (10, y), font, .8, (255, 255, 255), 3, cv2.LINE_8)

    cv2.imshow('Webcam Feed', filtered_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('g'):
        current_filter = 'g'
    elif key == ord('s'):
        current_filter = 's'
    elif key == ord('n'):
        current_filter = 'n'
    elif key == ord('b'):
        current_filter = 'b'
    elif key == ord('e'):
        current_filter = 'e'
    elif key == ord('o'):
        current_filter = 'o'
    elif key == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()