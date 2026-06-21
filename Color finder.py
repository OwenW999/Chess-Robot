import cv2
import numpy as np

cap = cv2.VideoCapture(0)

def show_hsv_at_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        hsv_frame = param
        pixel = hsv_frame[y, x]
        print(f"Clicked at ({x},{y}) -> HSV: {pixel}")

cv2.namedWindow("Click on board colors")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cv2.setMouseCallback("Click on board colors", show_hsv_at_click, hsv)
    cv2.imshow("Click on board colors", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()