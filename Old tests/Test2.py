import cv2

cap = cv2.VideoCapture(0)  # 0 = first camera (your webcam)

if not cap.isOpened():
    print("Error: Could not open camera.")
else:
    print("Camera opened successfully!")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        cv2.imshow("Webcam Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # press Q to quit
            break

cap.release()
cv2.destroyAllWindows()