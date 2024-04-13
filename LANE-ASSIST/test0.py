import cv
import numpy as np

def detect_lanes(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform edge detection using Canny
    edges = cv2.Canny(blurred, 50, 150)

    # Define region of interest (ROI)
    height, width = image.shape[:2]
    vertices = np.array([[(0, height), (width / 2, height / 2), (width, height)]], dtype=np.int32)
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Apply Hough transform to detect lines
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 15, np.array([]), minLineLength=10, maxLineGap=20)

    # Draw detected lanes on original image
    lane_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(lane_image, (x1, y1), (x2, y2), (0, 255, 0), 5)

    # Combine original image with lane lines
    result = cv2.addWeighted(image, 0.8, lane_image, 1, 0)

    return result

# Capture video from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect lanes
    lanes = detect_lanes(frame)

    # Display result
    cv2.imshow('Lane Assist', lanes)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
