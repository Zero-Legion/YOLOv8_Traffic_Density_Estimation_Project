import cv2
import numpy as np
from hailo_platform import HailoRT  # Hailo SDK
from picamera2 import Picamera2

# Load the compiled model for Hailo
HEF_PATH = "./HEF/yolo5m_vehicles.hef"  # If 'HEF' is in the current working directory

# Initialize Hailo model
hef = HailoRT.Hef(HEF_PATH)
runner = HailoRT.Runner(hef)

# Use Pi Camera with Picamera2 library
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (1280, 720)})
picam2.configure(config)
picam2.start()

# Define the threshold for heavy traffic
heavy_traffic_threshold = 10

# Define vertices for the quadrilaterals
vertices1 = np.array([(465, 350), (609, 350), (510, 630), (2, 630)], dtype=np.int32)
vertices2 = np.array([(678, 350), (815, 350), (1203, 630), (743, 630)], dtype=np.int32)

# Set vertical range and lane thresholds
x1, x2 = 325, 635
lane_threshold = 609

# Text annotation positions
text_position_left_lane = (10, 50)
text_position_right_lane = (820, 50)
intensity_position_left_lane = (10, 100)
intensity_position_right_lane = (820, 100)

# Font settings for text annotations
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (255, 255, 255)  # White
background_color = (0, 0, 255)  # Red

# Set up video writer to save the output
frame_width, frame_height = 1280, 720
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('processed_camera_feed.avi', fourcc, 20.0, (frame_width, frame_height))

# Main loop to capture and process frames
while True:
    # Capture frame-by-frame from Pi camera
    frame = picam2.capture_array()

    # Make a copy of the original frame to modify
    detection_frame = frame.copy()

    # Black out the regions outside the vertical range
    detection_frame[:x1, :] = 0  # Black out top
    detection_frame[x2:, :] = 0  # Black out bottom

    # Perform inference using Hailo SDK
    input_data = detection_frame.astype(np.float32) / 255.0
    input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension if required

    # Run inference
    output = runner.run(input_data)

    # Process inference results
    # Assuming the output gives bounding boxes similar to YOLO format
    processed_frame = detection_frame.copy()  # Modify this as per Hailo SDK's inference output

    # Draw quadrilaterals
    cv2.polylines(processed_frame, [vertices1], isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.polylines(processed_frame, [vertices2], isClosed=True, color=(255, 0, 0), thickness=2)

    # Initialize counters for left and right lane vehicles
    vehicles_in_left_lane = 0
    vehicles_in_right_lane = 0

    # Loop through bounding boxes and count vehicles in each lane
    for box in output:  # Adjust based on Hailo output format
        if box[0] < lane_threshold:
            vehicles_in_left_lane += 1
        else:
            vehicles_in_right_lane += 1

    # Determine traffic intensity
    traffic_intensity_left = "Heavy" if vehicles_in_left_lane > heavy_traffic_threshold else "Smooth"
    traffic_intensity_right = "Heavy" if vehicles_in_right_lane > heavy_traffic_threshold else "Smooth"

    # Add text annotations for left and right lanes
    cv2.rectangle(processed_frame, (text_position_left_lane[0]-10, text_position_left_lane[1]-25),
                  (text_position_left_lane[0]+460, text_position_left_lane[1]+10), background_color, -1)
    cv2.putText(processed_frame, f'Vehicles in Left Lane: {vehicles_in_left_lane}', text_position_left_lane,
                font, font_scale, font_color, 2, cv2.LINE_AA)
    
    cv2.rectangle(processed_frame, (intensity_position_left_lane[0]-10, intensity_position_left_lane[1]-25),
                  (intensity_position_left_lane[0]+460, intensity_position_left_lane[1]+10), background_color, -1)
    cv2.putText(processed_frame, f'Traffic Intensity: {traffic_intensity_left}', intensity_position_left_lane,
                font, font_scale, font_color, 2, cv2.LINE_AA)

    cv2.rectangle(processed_frame, (text_position_right_lane[0]-10, text_position_right_lane[1]-25),
                  (text_position_right_lane[0]+460, text_position_right_lane[1]+10), background_color, -1)
    cv2.putText(processed_frame, f'Vehicles in Right Lane: {vehicles_in_right_lane}', text_position_right_lane,
                font, font_scale, font_color, 2, cv2.LINE_AA)
    
    cv2.rectangle(processed_frame, (intensity_position_right_lane[0]-10, intensity_position_right_lane[1]-25),
                  (intensity_position_right_lane[0]+460, intensity_position_right_lane[1]+10), background_color, -1)
    cv2.putText(processed_frame, f'Traffic Intensity: {traffic_intensity_right}', intensity_position_right_lane,
                font, font_scale, font_color, 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Real-time Traffic Analysis', processed_frame)

    # Save the frame to video file
    out.write(processed_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources
picam2.stop()
out.release()
cv2.destroyAllWindows()
