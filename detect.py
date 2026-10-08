from ultralytics import YOLO
from datetime import datetime
import csv
import os

# Load YOLO model
model = YOLO("yolo11n.pt")

# Detect objects
results = model("test.jpg")

# Count detected objects
counts = {}

for result in results:
    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        if class_name in counts:
            counts[class_name] += 1
        else:
            counts[class_name] = 1

# Get current date and time
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

print("\nPHYS-X Detection Results")
print("-------------------------")
print("Time:", timestamp)

for object_name, count in counts.items():
    print(f"{object_name}: {count}")

# Save results to CSV
file_exists = os.path.exists("detection_log.csv")

with open("detection_log.csv", "a", newline="") as file:
    writer = csv.writer(file)

    if not file_exists:
        writer.writerow(["Timestamp", "Object", "Count"])

    for object_name, count in counts.items():
        writer.writerow([timestamp, object_name, count])

print("\nResults saved to detection_log.csv")
print("Detection completed!")