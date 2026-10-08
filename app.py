from flask import Flask, render_template, request, Response,send_file,redirect
from ultralytics import YOLO
from datetime import datetime
import os
import csv
import cv2

app = Flask(__name__)

# Load YOLO model
model = YOLO("yolo11n.pt")
CONFIDENCE_THRESHOLD = 0.50


# =========================
# LIVE CAMERA PAGE
# =========================

@app.route("/live")
def live():
    return render_template("live.html")


# =========================
# LIVE CAMERA GENERATOR
# =========================

def generate_frames():

    camera = cv2.VideoCapture(0)

    last_saved_second = None

    while True:

        success, frame = camera.read()

        if not success:
            break

        # Run YOLO detection
        results = model(frame)

        # Count detected objects
        counts = {}

        for result in results:

            for box in result.boxes:

                class_id = int(box.cls[0])
                confidence =  float(box.conf[0])

                class_name = model.names[class_id]
            if confidence >= CONFIDENCE_THRESHOLD:
                if class_name in counts:
                    counts[class_name] += 1
                else:
                    counts[class_name] = 1

        # Draw YOLO boxes
        annotated_frame = results[0].plot(conf=True)

        # =========================
        # DISPLAY TITLE
        # =========================

        cv2.putText(
            annotated_frame,
            "PHYS-X LIVE DETECTION",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        # =========================
        # DISPLAY OBJECT COUNTS
        # =========================

        y_position = 70

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

        # =========================
        # SAVE COUNTS TO CSV
        # =========================

        current_second = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        if counts and current_second != last_saved_second:

            csv_exists = os.path.exists(
                "detection_log.csv"
            )

            with open(
                "detection_log.csv",
                "a",
                newline=""
            ) as file:

                writer = csv.writer(file)

                if not csv_exists:

                    writer.writerow([
                        "Timestamp",
                        "Object",
                        "Count"
                    ])

                for object_name, count in counts.items():

                    writer.writerow([
                        current_second,
                        object_name,
                        count
                    ])

            last_saved_second = current_second

        # =========================
        # CONVERT FRAME TO JPEG
        # =========================

        success, buffer = cv2.imencode(
            ".jpg",
            annotated_frame
        )

        if not success:
            continue

        frame = buffer.tobytes()

        # Send frame to browser
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame
            + b"\r\n"
        )

    camera.release()


# =========================
# VIDEO FEED
# =========================

@app.route("/video_feed")
def video_feed():

    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# =========================
# MAIN DASHBOARD
# =========================
@app.route("/download")
def download():
    return send_file(
        "detection_log.csv",
        as_attachment=True,
        download_name="PHYS-X_Detection_Report.csv"
    )
@app.route("/clear")
def clear_history():

    if os.path.exists("detection_log.csv"):
        os.remove("detection_log.csv")

    return redirect("/")

@app.route("/", methods=["GET", "POST"])
def home():

    counts = {}
    timestamp = None
    result_image = None
    history = []

    total_records = 0
    total_objects = 0
    most_detected = "None"

    # =========================
    # IMAGE UPLOAD
    # =========================

    if request.method == "POST":

        image = request.files["image"]

        if image:

            os.makedirs("static", exist_ok=True)

            image_path = "static/uploaded_image.jpg"

            image.save(image_path)

            # Run YOLO detection
            results = model(image_path)

            # Save detected image
            results[0].save(
                filename="static/detected_image.jpg"
            )

            # Count objects
            for result in results:

                for box in result.boxes:

                    class_id = int(box.cls[0])

                    class_name = model.names[class_id]

                    if class_name in counts:
                        counts[class_name] += 1
                    else:
                        counts[class_name] = 1

            # Timestamp
            timestamp = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            # Save to CSV
            csv_exists = os.path.exists(
                "detection_log.csv"
            )

            with open(
                "detection_log.csv",
                "a",
                newline=""
            ) as file:

                writer = csv.writer(file)

                if not csv_exists:

                    writer.writerow([
                        "Timestamp",
                        "Object",
                        "Count"
                    ])

                for object_name, count in counts.items():

                    writer.writerow([
                        timestamp,
                        object_name,
                        count
                    ])

            result_image = "detected_image.jpg"

    # =========================
    # READ HISTORY
    # =========================

    if os.path.exists("detection_log.csv"):

        with open(
            "detection_log.csv",
            "r",
            newline=""
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:

                history.append(row)

    # =========================
    # DASHBOARD STATISTICS
    # =========================

    total_records = len(history)

    object_totals = {}

    for row in history:

        object_name = row["Object"]

        count = int(row["Count"])

        if object_name in object_totals:

            object_totals[object_name] += count

        else:

            object_totals[object_name] = count

    total_objects = sum(
        object_totals.values()
    )

    if object_totals:

        most_detected = max(
            object_totals,
            key=object_totals.get
        )

    # =========================
    # DISPLAY DASHBOARD
    # =========================

    return render_template(
        "index.html",
        counts=counts,
        timestamp=timestamp,
        result_image=result_image,
        history=history,
        total_records=total_records,
        total_objects=total_objects,
        most_detected=most_detected
    )


# =========================
# START FLASK
# =========================

if __name__ == "__main__":

    app.run(debug=True)