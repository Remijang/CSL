import cv2
import numpy as np
import time
from digit import get_prediction

# Choose your webcam: 0, 1, ...
cap = cv2.VideoCapture(1)

WIDTH = 640
HEIGHT = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 0.25 --> manual
cap.set(cv2.CAP_PROP_EXPOSURE, 2)  # exposure value (depends on the camera)

# other settings you need
def nothing(x):
    pass

cv2.namedWindow('Threshold Sliders')
cv2.createTrackbar('r', 'Threshold Sliders', 32, 255, nothing)
cv2.createTrackbar('g', 'Threshold Sliders', 64, 255, nothing)
cv2.createTrackbar('b', 'Threshold Sliders', 64, 255, nothing)

# Utils for drawing history
prev_center = None
line_history = []
history_duration = 60.0  # time limit

# Set frame per second to control sampling
target_fps = 60
frame_delay = 1.0 / target_fps
frame_cnt = 0
title_text = 'The number is ?'

def find_combined_bounded_square_white_regions(img):
    try:
        _, thresh = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        squares = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            side = max(w, h)
            x_square = x
            y_square = y
            if w > h:
                y_square = y - (w - h) // 2
            elif h > w:
                x_square = x - (h - w) // 2
            y_square = max(0, y_square)
            x_square = max(0, x_square)
            if x_square + side > img.shape[1]:
                side = img.shape[1] - x_square
            if y_square + side > img.shape[0]:
                side = img.shape[0] - y_square
            squares.append((x_square, y_square, side))

        if not squares:
            return None

        min_x = min(x for x, _, _ in squares)
        min_y = min(y for _, y, _ in squares)
        max_x = max(x + side for x, y, side in squares)
        max_y = max(y + side for x, y, side in squares)

        combined_side = max(max_x - min_x, max_y - min_y)

        return (min_x, min_y, combined_side)

    except Exception as e:
        print(f"Error: {e}")
        return None

while True:
    start_time = time.time()
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_b, frame_g, frame_r = cv2.split(frame)
    r_thres = cv2.getTrackbarPos('r', 'Threshold Sliders')
    g_thres = cv2.getTrackbarPos('g', 'Threshold Sliders')
    b_thres = cv2.getTrackbarPos('b', 'Threshold Sliders')
    ret, r = cv2.threshold(frame_r, r_thres, 255, cv2.THRESH_BINARY)
    ret, g_inv = cv2.threshold(frame_g, g_thres, 255, cv2.THRESH_BINARY_INV)
    ret, b_inv = cv2.threshold(frame_b, b_thres, 255, cv2.THRESH_BINARY_INV)
    result = cv2.bitwise_and(r, b_inv, mask=None)
    result = cv2.bitwise_and(result, g_inv, mask=None)
    result = cv2.blur(result, (3, 3))
    ret, result = cv2.threshold(result, 128, 255, cv2.THRESH_BINARY)
    ROI = np.array([[(80, 40), (540, 40), (540, 440), (80, 440)]])
    ROI_mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    cv2.fillPoly(ROI_mask, ROI, 255)
    result = cv2.bitwise_and(result, ROI_mask)
    contours, hierarchy = cv2.findContours(result, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    display = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        cv2.drawContours(display, largest_contour, -1, (0, 255, 0))
        if area > 256:
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
    current_time = time.time()
    valid_lines = []
    line_display = np.zeros_like(frame)
    for start, end, timestamp in line_history:
        if current_time - timestamp <= history_duration:
            cv2.line(line_display, start, end, (255, 255, 255), 20)
            valid_lines.append((start, end, timestamp))
    line_history = valid_lines
    gray_line_display = cv2.cvtColor(line_display, cv2.COLOR_BGR2GRAY)

    #find and draw combined bounded square
    combined_square = find_combined_bounded_square_white_regions(gray_line_display)
    if combined_square:
        x, y, side = combined_square
        cv2.rectangle(line_display, (x, y), (x + side, y + side), (0, 255, 0), 2)

        # Crop the line display to the rectangle
        cropped_line_display = gray_line_display[y:y + side, x:x + side]
        if cropped_line_display.size == 0:
            cropped_line_display = gray_line_display #if crop fails use whole image.
    else:
        cropped_line_display = gray_line_display #if no square, use whole image.

    combined_frame = cv2.hconcat([frame, display, line_display])
    if frame_cnt % 50 == 0:
        title_text = title_text[:14] + str(get_prediction(cropped_line_display))
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 50)
    fontScale = 1
    color = (0, 0, 255)
    thickness = 2
    combined_frame = cv2.putText(combined_frame, title_text, org, font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.imshow('Combined Frames', combined_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        line_history.clear()
    frame_cnt += 1
    end_time = time.time()
    process_time = end_time - start_time
    delay = max(frame_delay - process_time, 0.0)
    time.sleep(delay)
cap.release()
cv2.destroyAllWindows()