import cv2
import numpy as np

def remove_glare(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    glare_mask = cv2.inRange(hsv, (0, 0, 220), (180, 60, 255))
    glare_mask = cv2.dilate(glare_mask, None, iterations=2)
    result = cv2.inpaint(frame, glare_mask, 3, cv2.INPAINT_TELEA)
    return result

def process_for_board_detection(frame):
    deglared = remove_glare(frame)
    gray = cv2.cvtColor(deglared, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=11, C=2
    )
    return thresh

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Keep original color frame for display/drawing
    display_frame = frame.copy()

    # Get the processed (binary) version for contour detection
    processed = process_for_board_detection(frame)

    # Edges directly from the already-processed binary image —
    # no need to re-blur/re-threshold, that work is done
    edges = cv2.Canny(processed, 50, 150)
    edges = cv2.dilate(edges, None, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 5000:
            peri = cv2.arcLength(largest, True)
            approx = cv2.approxPolyDP(largest, 0.02 * peri, True)

            if len(approx) == 4:
                cv2.drawContours(display_frame, [approx], -1, (0, 255, 0), 3)

    cv2.imshow("Board Detection", display_frame)
    # Optional: see the intermediate processed image to debug
    cv2.imshow("Processed", processed)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()