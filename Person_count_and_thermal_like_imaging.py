import cv2
import numpy as np
from ultralytics import YOLO
# Initialize the YOLO model for person detection
model = YOLO('yolov8s.pt')
import supervision as sv

import torch
model = torch.hub.load('ultralytics/yolov5', 'yolov5x6')



# Initialize the default camera
cap = cv2.VideoCapture(0)

# Check if camera was opened successfully
if not cap.isOpened():
    print('Error: Could not open camera.')
else:
    # Set the resolution of the capture to the size of the display window
    width, height = 640, 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        # Check if the frame was captured successfully
        if not ret:
            print('Error: Could not capture frame.')
            break

        # Convert the captured frame to grayscale
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Compute the negative of the grayscale image
        img_neg = 255 - img_gray

        # Apply a color map to the negative image
        img_color = cv2.applyColorMap(img_neg, cv2.COLORMAP_JET)

        # Detect persons in the frame using the YOLO model
        results = model(frame, size=1280)
        detections = sv.Detections.from_yolov5(results)
        detections = detections[(detections.class_id == 0)]

        # Count the number of persons detected
        num_persons = len(detections)

        # Annotate the frame with the number of persons detected
        text_position = (int(0.05*width), int(0.1*height))
        text_scale = 1
        text_thickness = 1
        text_color = (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            img_color, 
            f"Person Count: {num_persons}", 
            text_position, 
            font, 
            text_scale, 
            text_color, 
            text_thickness
        )

        # Annotate the frame with bounding boxes around the detected persons
        box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)
        img_color = box_annotator.annotate(scene=img_color, detections=detections)

        # Resize the frame to the size of the display window
        img_color = cv2.resize(img_color, (width, height))

        # Display the annotated frame
        cv2.imshow('Thermal-like Image', img_color)

        # Break if 'q' key is pressed
        if cv2.waitKey(1) == ord('q'):
            break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
