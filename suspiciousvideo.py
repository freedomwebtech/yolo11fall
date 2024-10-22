import cv2
import numpy as np
from ultralytics import YOLO
import cvzone

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Load the YOLOv8 model
model = YOLO("yolo11s-seg.pt")
names = model.model.names

# Open the video file (or use a webcam, here using a video file)
cap = cv2.VideoCapture("fall.mp4")

# Resize dimensions
resize_width, resize_height = 1020, 500

# Define the codec and create a VideoWriter object to save the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
out = cv2.VideoWriter('output_blur_fall_detection.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (resize_width, resize_height))

count = 0

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    count += 1
    if count % 2 != 0:
        continue

    # Resize frame to match the output size
    frame = cv2.resize(frame, (resize_width, resize_height))

    # Run YOLOv8 tracking on the frame, persisting tracks between frames
    results = model.track(frame, persist=True, classes=0)

    # Ensure boxes exist in the results
    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.int().cpu().tolist()
        class_ids = results[0].boxes.cls.int().cpu().tolist()

        # Check if tracking IDs exist before attempting to retrieve them
        if results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()
        else:
            track_ids = [-1] * len(boxes)  # Use -1 for objects without IDs

        masks = results[0].masks
        if masks is not None:
            clss = results[0].boxes.cls.cpu().tolist()
            masks = masks.xy
            overlay = frame.copy()

            for box, track_id, class_id, mask in zip(boxes, track_ids, class_ids, masks):
                # Convert mask points to integer
                c = names[class_id]

                x1, y1, x2, y2 = box

                # Calculate height and width
                h = y2 - y1
                w = x2 - x1
                thresh = h - w
                print(thresh)

                # Extract the region of interest (ROI) from the frame
                roi = frame[y1:y2, x1:x2]

                # Check if a fall is detected (thresh <= 0)
                if thresh <= 0:
                    # Fall detected, remove blur (display clear)
                    frame[y1:y2, x1:x2] = overlay[y1:y2, x1:x2]  # Clear region
                    
                    # Add fall detection label and rectangle
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cvzone.putTextRect(frame, f'{track_id}', (x1, y2), 1, 1)
                    cvzone.putTextRect(frame, f"{'Fall'}", (x1, y1), 1, 1)

                else:
                    # Normal state, apply blur to the region
                    blur_obj = cv2.blur(roi, (50, 50))  # Apply blurring
                    frame[y1:y2, x1:x2] = blur_obj  # Replace the original region with blurred one
                    
                    # Add normal detection label and rectangle
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cvzone.putTextRect(frame, f'{track_id}', (x1, y2), 1, 1)
                    cvzone.putTextRect(frame, f"{'Normal'}", (x1, y1), 1, 1)

    # Write the processed frame to the output video file
    out.write(frame)

    # Show the frame
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture and writer objects and close the display window
cap.release()
out.release()
cv2.destroyAllWindows()
