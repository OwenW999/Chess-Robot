import cv2
import numpy as np

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)

tag_size = 200
margin = 50  # white border thickness on each side

for marker_id in range(4):
    tag_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, tag_size)

    # Create a white canvas bigger than the tag, then paste tag in the center
    canvas = np.full((tag_size + margin*2, tag_size + margin*2), 255, dtype=np.uint8)
    canvas[margin:margin+tag_size, margin:margin+tag_size] = tag_img

    cv2.imwrite(f"tag_{marker_id}_with_border.png", canvas)
    print(f"Saved tag_{marker_id}_with_border.png")