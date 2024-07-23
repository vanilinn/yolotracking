from collections import defaultdict
import cv2
import numpy as np
import time
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8s.pt")

# Open the RTSP video stream
rtsp_url = "rtsp_url"  # замените на ваш RTSP URL
cap = cv2.VideoCapture(rtsp_url)

# Check if the RTSP stream is opened correctly
if not cap.isOpened():
    print("Error: Could not open RTSP stream.")
    exit()

# Store the track history
track_history = defaultdict(lambda: [])

# Define the line parameters
line1_start = (300, 600)
line1_end = (1150, 600)
line2_start = (350, 350)
line2_end = (500, 270)
line_color1 = (0, 255, 0)  # Green color for the first line
line_color2 = (0, 0, 255)  # Red color for the second line
line_thickness = 3

# Sets to store IDs of tracks that have crossed each line
crossed_ids_line1 = set()
crossed_ids_line2 = set()

# Counters for objects crossing each line and both lines
counter1 = 0
counter2 = 0
counter_both = 0


# Function to check if line segment 'pt1-pt2' intersects with line segment 'ptA-ptB'
def intersects(pt1, pt2, ptA, ptB):
    def orientation(p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0
        elif val > 0:
            return 1
        else:
            return 2

    def on_segment(p, q, r):
        if min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[1] <= max(p[1], r[1]):
            return True
        return False

    p1, q1 = pt1, pt2
    p2, q2 = ptA, ptB

    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    if o1 != o2 and o3 != o4:
        return True

    if o1 == 0 and on_segment(p1, p2, q1):
        return True

    if o2 == 0 and on_segment(p1, q2, q1):
        return True

    if o3 == 0 and on_segment(p2, p1, q2):
        return True

    if o4 == 0 and on_segment(p2, q1, q2):
        return True

    return False


# Main processing loop
fps_limit = 24  # Частота кадров в секунду (fps)
frames_to_skip = 3  # Количество кадров для пропуска между обработками
frame_interval = 1.0 / fps_limit  # Интервал между кадрами

start_time = time.time()
frame_count = 0

while cap.isOpened():
    # current_time = time.time()
    # if current_time - start_time < frame_interval:
    #     time.sleep(frame_interval - (current_time - start_time))
    # start_time = time.time()

    # Read a frame from the RTSP stream
    success, frame = cap.read()

    if success:
        frame_count += 1

        # Пропускаем frames_to_skip кадров перед обработкой
        if frame_count % (frames_to_skip + 1) != 1:
            continue

        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)
        # print(results)

        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        # Visualize the results on the frame
        annotated_frame = results[0].plot(line_width=1, font_size=1)

        # Draw the lines on the frame
        cv2.line(annotated_frame, line1_start, line1_end, line_color1, line_thickness)
        cv2.line(annotated_frame, line2_start, line2_end, line_color2, line_thickness)

        # Check if objects cross each line
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            if len(track) > 150:  # retain 30 tracks for 30 frames
                track.pop(0)

            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=3)

            # Check if the track crosses each line
            if len(track) > 1:
                pt1 = (int(track[-2][0]), int(track[-2][1]))  # Previous point
                pt2 = (int(track[-1][0]), int(track[-1][1]))  # Current point

                # Check if the track crosses line 1
                if intersects(pt1, pt2, line1_start, line1_end):
                    if track_id not in crossed_ids_line1:
                        crossed_ids_line1.add(track_id)

                # Check if the track crosses line 2
                if intersects(pt1, pt2, line2_start, line2_end):
                    if track_id in crossed_ids_line1 and track_id not in crossed_ids_line2:
                        crossed_ids_line2.add(track_id)
                        counter_both += 1
                        crossed_ids_line1.remove(track_id)
                    elif track_id not in crossed_ids_line1 and track_id not in crossed_ids_line2:
                        counter2 += 1

        # Display the counters
        cv2.putText(annotated_frame, f"Counter Both Lines: {counter_both}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 255), 2)
        cv2.putText(annotated_frame, f"Counter 1: {counter1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 255), 2)
        cv2.putText(annotated_frame, f"Counter 2: {counter2}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 255), 2)

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
