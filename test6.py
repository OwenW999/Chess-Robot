# # import cv2
# # import numpy as np
# # from collections import deque

# # cap = cv2.VideoCapture(0)
# # corner_history = deque(maxlen=10)  # average over last 10 frames

# # while True:
# #     ret, frame = cap.read()
# #     if not ret:
# #         break

# #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# #     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# #     edges = cv2.Canny(blurred, 50, 150)
# #     edges = cv2.dilate(edges, None, iterations=2)

# #     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# #     display = frame.copy()
# #     if contours:
# #         largest = max(contours, key=cv2.contourArea)
# #         if cv2.contourArea(largest) > 5000:
# #             peri = cv2.arcLength(largest, True)
# #             approx = cv2.approxPolyDP(largest, 0.02 * peri, True)

# #             if len(approx) == 4:
# #                 corner_history.append(approx.reshape(4, 2))

# #                 # Average corner positions across recent frames
# #                 avg_corners = np.mean(corner_history, axis=0).astype(int)

# #                 cv2.drawContours(display, [avg_corners.reshape(4, 1, 2)], -1, (0, 255, 0), 3)
# #                 for (x, y) in avg_corners:
# #                     cv2.circle(display, (x, y), 6, (0, 0, 255), -1)

# #     cv2.imshow("Stabilized Detection", display)

# #     if cv2.waitKey(1) & 0xFF == ord('q'):
# #         break

# # cap.release()
# # cv2.destroyAllWindows()

# import cv2
# import numpy as np
# from collections import deque

# cap = cv2.VideoCapture(0)
# corner_history = deque(maxlen=10)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     edges = cv2.Canny(blurred, 80, 150)
#     edges = cv2.dilate(edges, None, iterations=2)

#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     display = frame.copy()
#     debug = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

#     if contours:
#         largest = max(contours, key=cv2.contourArea)
#         # Draw the RAW largest contour in blue, before approximation
#         cv2.drawContours(debug, [largest], -1, (255, 0, 0), 2)

#         if cv2.contourArea(largest) > 5000:
#             peri = cv2.arcLength(largest, True)
#             approx = cv2.approxPolyDP(largest, 0.02 * peri, True)

#             if len(approx) == 4:
#                 corner_history.append(approx.reshape(4, 2))
#                 avg_corners = np.mean(corner_history, axis=0).astype(int)
#                 cv2.drawContours(display, [avg_corners.reshape(4, 1, 2)], -1, (0, 255, 0), 3)
#                 for (x, y) in avg_corners:
#                     cv2.circle(display, (x, y), 6, (0, 0, 255), -1)
#             else:
#                 cv2.putText(display, f"Found {len(approx)} corners, not 4", (10, 30),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

#     cv2.imshow("Result", display)
#     cv2.imshow("Raw Contour (blue) on Edges", debug)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

import cv2
import numpy as np
from collections import deque

cap = cv2.VideoCapture(0)
corner_history = deque(maxlen=10)

# Tuned to your wood/border samples
lower_wood = np.array([0, 90, 40])
upper_wood = np.array([18, 255, 160])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_wood, upper_wood)

    # Clean up noise, then close gaps so the border forms a connected shape
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

    # Find contours directly on the mask (no Canny needed - mask IS the binary image)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    display = frame.copy()

    if contours:
        largest = max(contours, key=cv2.contourArea)

        if cv2.contourArea(largest) > 5000:
            hull = cv2.convexHull(largest)
            peri = cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, 0.02 * peri, True)

            if len(approx) == 4:
                corner_history.append(approx.reshape(4, 2))
                avg_corners = np.mean(corner_history, axis=0).astype(int)
                cv2.drawContours(display, [avg_corners.reshape(4, 1, 2)], -1, (0, 255, 0), 3)
                for (x, y) in avg_corners:
                    cv2.circle(display, (x, y), 6, (0, 0, 255), -1)
            else:
                cv2.drawContours(display, [hull], -1, (255, 0, 0), 2)
                cv2.putText(display, f"{len(approx)} corners found", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Result", display)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()