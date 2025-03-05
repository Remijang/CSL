import cv2
import numpy as np

# Choose your webcam: 0, 1, ...
cap = cv2.VideoCapture(0)

WIDTH = 640
HEIGHT = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 0.25 --> manual
cap.set(cv2.CAP_PROP_EXPOSURE, -1)  # exposure value (depends on the camera)

# other settings you need
def nothing(x):
	pass
cv2.namedWindow('Threshold Sliders')
cv2.createTrackbar('r','Threshold Sliders',24,255,nothing)
cv2.createTrackbar('g','Threshold Sliders',71,255,nothing)
cv2.createTrackbar('b','Threshold Sliders',74,255,nothing)


while(True):
	# Get one frame from the camera
	ret, frame = cap.read()

	# Turn the frame into grayscale
	frame_b, frame_g, frame_r = cv2.split(frame)
	
	r_thres = cv2.getTrackbarPos('r', 'Threshold Sliders')
	g_thres = cv2.getTrackbarPos('g', 'Threshold Sliders')
	b_thres = cv2.getTrackbarPos('b', 'Threshold Sliders')
	
	ret, r = cv2.threshold(frame_r, r_thres, 255, cv2.THRESH_BINARY)
	ret, g_inv = cv2.threshold(frame_g, g_thres, 255, cv2.THRESH_BINARY_INV)
	ret, b_inv = cv2.threshold(frame_b, b_thres, 255, cv2.THRESH_BINARY_INV)
	
	result = cv2.bitwise_and(r, b_inv, mask = None)
	result = cv2.bitwise_and(result, g_inv, mask = None)
	result = cv2.blur(result, (5, 5))
	ret, result = cv2.threshold(result, 128, 255, cv2.THRESH_BINARY)
	
	contours, hierarchy = cv2.findContours(result, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	display = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

	# Perform thresholding


	# Get the final result using bitwise operations, like ROI masks (optional)


	# Find and draw contours


	# Iterate through each contour, check the area and find the center


	# Show the frame
	cv2.imshow('frame', display)

	# Press q to quit
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# Release the camera
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
