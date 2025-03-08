import cv2
import numpy as np
import time
from digit import get_prediction
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
cv2.createTrackbar('r', 'Threshold Sliders', 64, 255, nothing)
cv2.createTrackbar('g', 'Threshold Sliders', 64, 255, nothing)
cv2.createTrackbar('b', 'Threshold Sliders', 64, 255, nothing)

# Utils for drawing history
prev_center = None
line_history = []
history_duration = 5.0  # time limit

# Set frame per second to control sampling
target_fps = 30
frame_delay = 1.0 / target_fps
frame_cnt = 0
title_text = 'The number is ?'
while True:
    start_time = time.time()

    # Get one frame from the camera
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    # Turn the frame into grayscale
    frame_b, frame_g, frame_r = cv2.split(frame)

    r_thres = cv2.getTrackbarPos('r', 'Threshold Sliders')
    g_thres = cv2.getTrackbarPos('g', 'Threshold Sliders')
    b_thres = cv2.getTrackbarPos('b', 'Threshold Sliders')

    # Perform thresholding
    ret, r = cv2.threshold(frame_r, r_thres, 255, cv2.THRESH_BINARY)
    ret, g_inv = cv2.threshold(frame_g, g_thres, 255, cv2.THRESH_BINARY_INV)
    ret, b_inv = cv2.threshold(frame_b, b_thres, 255, cv2.THRESH_BINARY_INV)

    # Get the final result using bitwise operations, like ROI masks (optional)
    result = cv2.bitwise_and(r, b_inv, mask=None)
    result = cv2.bitwise_and(result, g_inv, mask=None)
    result = cv2.blur(result, (5, 5))
    ret, result = cv2.threshold(result, 128, 255, cv2.THRESH_BINARY)

    # Find and draw contours
    contours, hierarchy = cv2.findContours(result, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    display = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    # Iterate through each contour, check the area and find the center
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        cv2.drawContours(display, largest_contour, -1, (0, 255, 0))

        # Set the threshold for removing noise
        if area > 256:

            # Use moment function to find the center
            M = cv2.moments(largest_contour)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            
            cv2.circle(display, (cX, cY), 20, (0, 0, 255), -1)
            if prev_center is not None:
                line_history.append((prev_center, (cX, cY), time.time()))
            prev_center = (cX, cY)
        else:
            prev_center = None
    else:
        prev_center = None

	# Draw the lines
    current_time = time.time()
    valid_lines = []
    line_display = np.zeros_like(frame)
    for start, end, timestamp in line_history:
        if current_time - timestamp <= history_duration:
            cv2.line(line_display, start, end, (255, 255, 255), 20)
            valid_lines.append((start, end, timestamp))
    line_history = valid_lines
    
    # Convert it to grayscale image (640x480, 8bit)
    gray_line_display = cv2.cvtColor(line_display, cv2.COLOR_BGR2GRAY)

    # Use cv2.hconcat to concatenate frames horizontally
    combined_frame = cv2.hconcat([display, line_display])
    if frame_cnt %150 == 0:
        title_text = title_text[:14] + str(get_prediction(gray_line_display))
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 50)
    fontScale = 1
    color = (0, 0, 255)
    thickness = 2
    combined_frame = cv2.putText(combined_frame, title_text, org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
    # Show the combined frame
    cv2.imshow('Combined Frames', combined_frame)
    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    frame_cnt += 1
    
	# Fps control
    end_time = time.time()
    process_time = end_time - start_time
    delay = max(frame_delay - process_time, 0.0)
    time.sleep(delay)

# Release the camera
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()