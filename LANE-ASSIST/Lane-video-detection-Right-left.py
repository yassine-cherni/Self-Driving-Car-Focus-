import cv2
import numpy as np

# Define lane departure threshold
threshold = 50  # Adjust this value based on image resolution and desired sensitivity

def detect_lanes(frame):
    # Preprocess the frame (grayscale conversion, noise reduction)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 100, 200)  # Adjust Canny parameters for better edge detection

    # Apply Hough transform to identify lane lines
    lines = cv2.HoughLinesP(canny, rho=1, theta=np.pi/180, threshold=50, minLineLength=50, maxLineGap=30)
    left_lane = []
    right_lane = []

    # Separate left and right lane lines based on their angle and position
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 != 0:  # Check for division by zero
                slope = (y2 - y1) / (x2 - x1)
                if slope < -0.5:  # Left lane line
                    left_lane.append([x1, y1, x2, y2])
                elif slope > 0.5:  # Right lane line
                    right_lane.append([x1, y1, x2, y2])

    # Choose the best fitting line for each side (optional)
    if left_lane:
        left_lane = np.average(left_lane, axis=0).astype(int)
    if right_lane:
        right_lane = np.average(right_lane, axis=0).astype(int)

    return left_lane, right_lane

def calculate_distance(lane_line, reference_position):
    # Assuming lane line is represented as [x1, y1, x2, y2]
    if lane_line is None or len(lane_line) == 0:
        return 0  # No lane detected

    # Calculate line equation coefficients
    slope = (lane_line[3] - lane_line[1]) / (lane_line[2] - lane_line[0])
    y_intercept = lane_line[1] - slope * lane_line[0]

    # Calculate distance from lane center (reference_position) to the line
    distance = abs(slope * reference_position - lane_line[1] + y_intercept) / np.sqrt(slope**2 + 1)

    # Positive distance indicates car is right of the lane, negative indicates left
    return distance

def lane_assist(frame):
    left_lane, right_lane = detect_lanes(frame)
    center_position = frame.shape[1] // 2

    # Calculate distances to left and right lanes
    left_distance = calculate_distance(left_lane, center_position)
    right_distance = calculate_distance(right_lane, center_position)

    # Display lane departure warning based on distances
    warning_message = ""
    if left_distance > threshold:
        warning_message = "Lane Departure: Left!"
    elif right_distance > threshold:
        warning_message = "Lane Departure: Right!"

    # Draw lane lines and center position (optional for visualization)
    if left_lane.size > 0:
        cv2.line(frame, (left_lane[0], left_lane[1]), (left_lane[2], left_lane[3]), (0, 255, 0), 3)
    if right_lane.size > 0:
        cv2.line(frame, (right_lane[0], right_lane[1]), (right_lane[2], right_lane[3]), (0, 0, 255), 3)
    cv2.line(frame, (center_position, 0), (center_position, frame.shape[0]), (255, 0, 0), 2)

    # Display warning message (optional)
    if warning_message:
        cv2.putText(frame, warning_message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return frame

# Open video file
video_path = "Lane2.mp4"  # Replace with the path to your video file
cap = cv2.VideoCapture(video_path)

# Define output video parameters
output_width = 640
output_height = 360
output_path = "lane_assist_output.mp4"
fps = cap.get(cv2.CAP_PROP_FPS)
codec = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter(output_path, codec, fps, (output_width, output_height))

# Process each frame of the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for faster processing (optional)
    frame = cv2.resize(frame, (output_width, output_height))

    # Apply lane assist function to the frame
    processed_frame = lane_assist(frame)

    # Write processed frame to output video
    output_video.write(processed_frame)

    # Display processed frame (optional)
    cv2.imshow('Lane Assist', processed_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video objects
cap.release()
output_video.release()
cv2.destroyAllWindows()
