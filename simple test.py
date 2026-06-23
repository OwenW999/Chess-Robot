import cv2

img = cv2.imread("raw_frames/frame_0021.png")
print("Image loaded:", img is not None, "shape:", None if img is None else img.shape)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
detector_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

corners, ids, rejected = detector.detectMarkers(img)
print("Detected IDs:", ids)
print("Number of rejected candidates:", len(rejected) if rejected is not None else 0)