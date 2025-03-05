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


while(True):
	# Get one frame from the camera
	ret, frame = cap.read()

	# Turn the frame into grayscale


	# Perform thresholding


	# Get the final result using bitwise operations, like ROI masks (optional)


	# Find and draw contours


	# Iterate through each contour, check the area and find the center


	# Show the frame
	cv2.imshow('frame', frame)

	# Press q to quit
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# Release the camera
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
