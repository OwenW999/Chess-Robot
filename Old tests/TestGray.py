import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    glare_mask = cv2.inRange(hsv, (0, 0, 230), (180, 40, 255))  # narrower, stricter glare definition
    glare_mask = cv2.dilate(glare_mask, None, iterations=1)

    deglared = cv2.inpaint(frame, glare_mask, 5, cv2.INPAINT_TELEA)

    cv2.imshow("Original", frame)
    cv2.imshow("Glare Mask", glare_mask)
    cv2.imshow("Deglared", deglared)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()