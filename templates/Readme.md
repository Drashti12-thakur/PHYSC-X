# PHYS-X — Smart Environment Monitoring Using AI and Computer Vision

PHYS-X is an AI-powered smart environment monitoring system that uses Computer Vision and YOLO object detection to identify and count objects from images and live camera feeds.

The system can detect objects, display confidence scores, maintain detection history, and generate a downloadable detection report.

---

## 🚀 Features

- 🖼️ Image-based object detection
- 📷 Real-time webcam monitoring
- 🤖 YOLO-based object detection
- 🔢 Automatic object counting
- 🎯 Confidence score display
- 📊 Confidence threshold filtering
- 🕐 Timestamp-based detection logging
- 📈 Detection statistics and charts
- 📄 Detection history
- 📥 Downloadable CSV detection report
- 🗑️ Clear detection history
- 🌐 Web-based dashboard

---

## 🛠️ Technologies Used

- Python
- Flask
- YOLO
- Ultralytics
- OpenCV
- HTML
- CSS
- JavaScript
- CSV

---

## ⚙️ System Workflow

```text
Camera / Image
      ↓
   OpenCV
      ↓
YOLO Object Detection
      ↓
Confidence Filtering
      ↓
Object Counting
      ↓
Timestamp
      ↓
CSV Logging
      ↓
Flask Web Dashboard
      ↓
Charts / History / Report