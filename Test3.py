import cv2

cap = cv2.VideoCapture(0)

# Set this to match your actual chessboard's INNER corners
# e.g. a standard 8x8 chessboard has 7x7 inner corners
CHESSBOARD_SIZE = (7, 7)

if not cap.isOpened():
    print("Error: Could not open camera.")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Try to find the chessboard corners
        found, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

        if found:
            # Draw the detected corners on the frame
            cv2.drawChessboardCorners(frame, CHESSBOARD_SIZE, corners, found)
            cv2.putText(frame, "Chessboard FOUND", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Chessboard NOT found", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Chessboard Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()