import cv2
import numpy as np

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
detector_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

cap = cv2.VideoCapture(0)

# Map tag ID -> role
TAG_ROLES = {0: "top-left", 1: "top-right", 2: "bottom-right", 3: "bottom-left"}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    corners, ids, rejected = detector.detectMarkers(frame)

    display = frame.copy()
    board_corners = {}

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(display, corners, ids)

        for i, tag_id in enumerate(ids.flatten()):
            if tag_id in TAG_ROLES:
                # Center of the tag = average of its 4 detected corners
                tag_corners = corners[i][0]
                center = tag_corners.mean(axis=0)
                board_corners[TAG_ROLES[tag_id]] = center

    # Only proceed if all 4 tags are visible
    if len(board_corners) == 4:
        pts = np.array([
            board_corners["top-left"],
            board_corners["top-right"],
            board_corners["bottom-right"],
            board_corners["bottom-left"]
        ], dtype="float32")

        # Draw the board outline using the 4 tag centers
        cv2.polylines(display, [pts.astype(int)], isClosed=True, color=(0, 255, 0), thickness=3)

        # This is your perspective transform, ready to use:
        size = 800
        dst = np.array([[0, 0], [size, 0], [size, size], [0, size]], dtype="float32")
        matrix = cv2.getPerspectiveTransform(pts, dst)
        warped = cv2.warpPerspective(frame, matrix, (size, size))
        cv2.imshow("Warped Board", warped)

    cv2.imshow("Detection", display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()