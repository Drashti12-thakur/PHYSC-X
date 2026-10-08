import cv2
from ultralytics import YOLO
from datetime import datetime
import csv
import os

# Load YOLO model
model = YOLO("yolo11n.pt")

# Open camera
camera = cv2.VideoCapture(0)

# Prevent saving every camera frame
last_saved_second = None

while True:

    ret, frame = camera.read()

    if not ret:
        print("Could not access camera")
        break

    # Run YOLO detection
    results = model(frame)

    # Count objects
    counts = {}

    for result in results:

        for box in result.boxes:

            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            if class_name in counts:
                counts[class_name] += 1
            else:
                counts[class_name] = 1

    # Draw detection boxes
    annotated_frame = results[0].plot()

    # Display title
    cv2.putText(
        annotated_frame,
        "PHYS-X LIVE DETECTION",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # Display counts
    y_position = 65

    for object_name, count in counts.items():

        text = f"{object_name}: {count}"

        cv2.putText(
            annotated_frame,
            text,
            (20, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        y_position += 30

    # Save counts once per second
    current_second = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if counts and current_second != last_saved_second:

        csv_exists = os.path.exists("detection_log.csv")

        with open("detection_log.csv", "a", newline="") as file:

            writer = csv.writer(file)

            if not csv_exists:
                writer.writerow(["Timestamp", "Object", "Count"])

            for object_name, count in counts.items():

                writer.writerow([
                    current_second,
                    object_name,
                    count
                ])

        last_saved_second = current_second

    # Show live camera
    cv2.imshow("PHYS-X Live Detection", annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()