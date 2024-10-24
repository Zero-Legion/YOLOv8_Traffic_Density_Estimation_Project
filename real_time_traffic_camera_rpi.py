import cv2
import numpy as np
from ultralytics import YOLO
from picamera import PiCamera
from picamera.array import PiRGBArray
import time

# Load the best fine-tuned YOLOv8 model
best_model = YOLO('models/best.pt')

# Define the threshold for considering traffic as heavy
heavy_traffic_threshold = 10

# Define the vertices for the quadrilaterals
vertices1 = np.array([(465, 350), (609, 350), (510, 630), (2, 630)], dtype=np.int32)
vertices2 = np.array([(678, 350), (815, 350), (1203, 630), (743, 630)], dtype=np.int32)

# Define the vertical range for the slice and lane threshold
x1, x2 = 325, 635
lane_threshold = 609

# Define the positions for the text annotations on the image
text_position_left_lane = (10, 50)
text_position_right_lane = (820, 50)
intensity_position_left_lane = (10, 100)
intensity_position_right_lane = (820, 100)

# Define font, scale, and colors for the annotations
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (255, 255, 255)  # White color for text
background_color = (0, 0, 255)  # Red background for text

# Initialize the Raspberry Pi camera
camera = PiCamera()
camera.resolution = (1280, 720)
camera.framerate = 20
raw_capture = PiRGBArray(camera, size=(1280, 720))

# Allow the camera to warm up
time.sleep(0.1)

# Define the codec and create a VideoWriter object for saving the output video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('processed_camera_feed.avi', fourcc, 20.0, (1280, 720))

# Read frames from the camera
for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
    image = frame.array

    # Create a copy of the original frame to modify
    detection_frame = image.copy()

    # Black out the regions outside the specified vertical range
    detection_frame[:x1, :] = 0  # Black out from top to x1
    detection_frame[x2:, :] = 0  # Black out from x2 to the bottom of the frame

    # Perform inference on the modified frame
    results = best_model.predict(detection_frame, imgsz=640, conf=0.4)
    processed_frame = results[0].plot(line_width=1)

    # Restore the original top and bottom parts of the frame
    processed_frame[:x1, :] = image[:x1, :].copy()
    processed_frame[x2:, :] = image[x2:, :].copy()

    # Draw the quadrilaterals on the processed frame
    cv2.polylines(processed_frame, [vertices1], isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.polylines(processed_frame, [vertices2], isClosed=True, color=(255, 0, 0), thickness=2)

    # Retrieve the bounding boxes from the results
    bounding_boxes = results[0].boxes

    # Initialize counters for vehicles in each lane
    vehicles_in_left_lane = 0
    vehicles_in_right_lane = 0

    # Loop through each bounding box to count vehicles in each lane
    for box in bounding_boxes.xyxy:
        # Check if the vehicle is in the left lane based on the x-coordinate of the bounding box
        if box[0] < lane_threshold:
            vehicles_in_left_lane += 1
        else:
            vehicles_in_right_lane += 1

    # Determine the traffic intensity for both lanes
    traffic_intensity_left = "Heavy" if vehicles_in_left_lane > heavy_traffic_threshold else "Smooth"
    traffic_intensity_right = "Heavy" if vehicles_in_right_lane > heavy_traffic_threshold else "Smooth"

    # Add text and background rectangles for vehicle counts and traffic intensity
    for text_position, vehicle_count, intensity_position, intensity_text in [
        (text_position_left_lane, vehicles_in_left_lane, intensity_position_left_lane, traffic_intensity_left),
        (text_position_right_lane, vehicles_in_right_lane, intensity_position_right_lane, traffic_intensity_right)
    ]:
        # Vehicle count rectangle and text
        cv2.rectangle(processed_frame, (text_position[0]-10, text_position[1] - 25), 
                      (text_position[0] + 460, text_position[1] + 10), background_color, -1)
        cv2.putText(processed_frame, f'Vehicles: {vehicle_count}', text_position, 
                    font, font_scale, font_color, 2, cv2.LINE_AA)

        # Traffic intensity rectangle and text
        cv2.rectangle(processed_frame, (intensity_position[0]-10, intensity_position[1] - 25), 
                      (intensity_position[0] + 460, intensity_position[1] + 10), background_color, -1)
        cv2.putText(processed_frame, f'Traffic Intensity: {intensity_text}', intensity_position, 
                    font, font_scale, font_color, 2, cv2.LINE_AA)

    # Display the processed frame
    cv2.imshow('Real-time Traffic Analysis', processed_frame)

    # Write the frame to the output video
    out.write(processed_frame)

    # Clear the stream for the next frame
    raw_capture.truncate(0)

    # Press Q on keyboard to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and video write objects
camera.close()
out.release()
cv2.destroyAllWindows()
