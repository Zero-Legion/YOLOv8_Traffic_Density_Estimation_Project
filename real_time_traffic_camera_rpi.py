import cv2
import numpy as np
from ultralytics import YOLO
import subprocess
import sys

# Load the best fine-tuned YOLOv8 model
best_model = YOLO('models/best.pt')

# Other code remains the same...

# Use libcamera to capture video
command = ['libcamera-vid', '--width', '1280', '--height', '720', '--framerate', '30', '--inline', '--output', '/dev/stdout']
process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# Check if the process started successfully
if process.poll() is not None:
    print("Error starting libcamera-vid:", process.stderr.read())
    sys.exit(1)

# Read until the camera feed is open
while True:
    raw_frame = process.stdout.read(1280 * 720 * 3)  # Read raw image data
    if not raw_frame:
        break  # Exit if no more frames

    # Convert the raw frame to a numpy array
    frame = np.frombuffer(raw_frame, np.uint8).reshape((720, 1280, 3))

    # Other processing code remains the same...

# Clean up
process.terminate()
cv2.destroyAllWindows()
