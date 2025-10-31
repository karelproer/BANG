import cv2
import numpy as np
import mouse
import time

# Define yellow color range in HSV
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])

# Small delay to avoid too many clicks
click_delay = 1

# Open webcam
cap = cv2.VideoCapture(0)  # 0 = default camera

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create mask for yellow
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Find coordinates of yellow pixels
    yellow_pixels = np.column_stack(np.where(mask > 0))
    
    if yellow_pixels.size > 0:
        # Take the first yellow pixel
        y, x = yellow_pixels[0]
        # Click relative to screen (optional: adjust if using window)
        mouse.click("left")
        print(f"Clicked at ({x}, {y})")
        time.sleep(click_delay)
    
    # Optional: show camera and mask
    cv2.imshow("Camera", frame)
    cv2.imshow("Mask", mask)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
