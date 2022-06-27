import numpy as np
import cv2
from collections import deque


# Default called trackbar function:
def setValues(x):
    print("")


# Different arrays for colour points of different colour:
blue_points = [deque(maxlen=1024)]
green_points = [deque(maxlen=1024)]
red_points = [deque(maxlen=1024)]
yellow_points = [deque(maxlen=1024)]

# Indexes to mark the points in particular arrays of specific colour:
blue_index = 0
green_index = 0
red_index = 0
yellow_index = 0

# Trackbars for adjusting marker colour:
cv2.namedWindow("Color detectors")
cv2.createTrackbar("Upper Hue", "Color detectors", 153, 180, setValues)
cv2.createTrackbar("Upper Saturation", "Color detectors", 255, 255, setValues)
cv2.createTrackbar("Upper Value", "Color detectors", 255, 255, setValues)
cv2.createTrackbar("Lower Hue", "Color detectors", 64, 180, setValues)
cv2.createTrackbar("Lower Saturation", "Color detectors", 72, 255, setValues)
cv2.createTrackbar("Lower Value", "Color detectors", 49, 255, setValues)

# The kernel to be used for dilation purpose:
kernel = np.ones((5, 5), np.uint8)

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
color_Index = 0

# Code for airpad setup:
window_Paint = np.zeros((471, 636, 3)) + 255
window_Paint = cv2.rectangle(window_Paint, (40, 1), (140, 65), (0, 0, 0), 2)
window_Paint = cv2.rectangle(window_Paint, (160, 1), (255, 65), colors[0], -1)
window_Paint = cv2.rectangle(window_Paint, (275, 1), (370, 65), colors[1], -1)
window_Paint = cv2.rectangle(window_Paint, (390, 1), (485, 65), colors[2], -1)
window_Paint = cv2.rectangle(window_Paint, (505, 1), (600, 65), colors[3], -1)

cv2.putText(window_Paint, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(window_Paint, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(window_Paint, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(window_Paint, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(window_Paint, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 2, cv2.LINE_AA)
cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

# Loading default webcam of PC:
cap = cv2.VideoCapture(0)


while True:
    # Reading frame from camera:
    ret, frame = cap.read()
    # Flipping frame to see same side:
    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    hue_u = cv2.getTrackbarPos("Upper Hue", "Color detectors")
    saturation_u = cv2.getTrackbarPos("Upper Saturation", "Color detectors")
    value_u = cv2.getTrackbarPos("Upper Value", "Color detectors")
    hue_l = cv2.getTrackbarPos("Lower Hue", "Color detectors")
    saturation_l = cv2.getTrackbarPos("Lower Saturation", "Color detectors")
    value_l = cv2.getTrackbarPos("Lower Value", "Color detectors")
    hsv_Upper = np.array([hue_u, saturation_u, value_u])
    hsv_Lower = np.array([hue_l, saturation_l, value_l])

    # Adding colour buttons to the live frame for colour access:
    frame = cv2.rectangle(frame, (40, 1), (140, 65), (122, 122, 122), -1)
    frame = cv2.rectangle(frame, (160, 1), (255, 65), colors[0], -1)
    frame = cv2.rectangle(frame, (275, 1), (370, 65), colors[1], -1)
    frame = cv2.rectangle(frame, (390, 1), (485, 65), colors[2], -1)
    frame = cv2.rectangle(frame, (505, 1), (600, 65), colors[3], -1)
    cv2.putText(frame, "CLEAR ALL", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 2, cv2.LINE_AA)

    # Identifying pointer by making mask:
    mask = cv2.inRange(hsv, hsv_Lower, hsv_Upper)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Contours for the pointer after identifying it:
    cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    center = None

    # If the contours are formed:
    if len(cnts) > 0:
        # Sorting contours to find biggest:
        cnt = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
        # Radius of the circle around the contour:
        ((x, y), radius) = cv2.minEnclosingCircle(cnt)
        # Drawing circle around the contour:
        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
        # Calculating the center of the detected contour:
        M = cv2.moments(cnt)
        center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

        # Checking for user to click on any button above the screen:
        if center[1] <= 65:
            if 40 <= center[0] <= 140:  # Clear Button
                blue_points = [deque(maxlen=512)]
                green_points = [deque(maxlen=512)]
                red_points = [deque(maxlen=512)]
                yellow_points = [deque(maxlen=512)]

                blue_index = 0
                green_index = 0
                red_index = 0
                yellow_index = 0

                window_Paint[67:, :, :] = 255
            elif 160 <= center[0] <= 255:
                color_Index = 0  # Blue
            elif 275 <= center[0] <= 370:
                color_Index = 1  # Green
            elif 390 <= center[0] <= 485:
                color_Index = 2  # Red
            elif 505 <= center[0] <= 600:
                color_Index = 3  # Yellow
        else:
            if color_Index == 0:
                blue_points[blue_index].appendleft(center)
            elif color_Index == 1:
                green_points[green_index].appendleft(center)
            elif color_Index == 2:
                red_points[red_index].appendleft(center)
            elif color_Index == 3:
                yellow_points[yellow_index].appendleft(center)
    # Append the next deques when nothing is detected to avois messing up
    else:
        blue_points.append(deque(maxlen=512))
        blue_index += 1
        green_points.append(deque(maxlen=512))
        green_index += 1
        red_points.append(deque(maxlen=512))
        red_index += 1
        yellow_points.append(deque(maxlen=512))
        yellow_index += 1

    # Draw lines of all the colors on the canvas and frame
    points = [blue_points, green_points, red_points, yellow_points]
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                cv2.line(window_Paint, points[i][j][k - 1], points[i][j][k], colors[i], 2)

    # Show all the windows
    cv2.imshow("Tracking", frame)
    cv2.imshow("Paint", window_Paint)
    cv2.imshow("mask", mask)

    # If the 'q' key is pressed then stop the application
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the camera and all resources
cap.release()
cv2.destroyAllWindows()
