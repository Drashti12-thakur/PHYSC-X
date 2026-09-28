import os
from ultralytics import YOLO
from fastapi import FastAPI, UploadFile, File
from pymongo import MongoClient
from dotenv import load_dotenv
from urllib.parse import quote_plus

load_dotenv()

app = FastAPI()
model = YOLO("yolo11n.pt")


username = "tdrishtii06_db_user"
password = os.getenv("MONGO_PASSWORD")


encoded_password = quote_plus(password)

MONGODB_URL = (
    f"mongodb+srv://{username}:{encoded_password}"
    "@cluster0.2wghxvt.mongodb.net/?appName=Cluster0"
)

client = MongoClient(MONGODB_URL)

db = client["physical_world_ai"]
users_collection = db["users"]
detections_collection = db["detections"]
@app.post("/detect")
async def detect_image(file: UploadFile = File(...)):
    contents = await file.read()

    temp_file = "temp.jpg"

    with open(temp_file, "wb") as f:
        f.write(contents)

    results = model(temp_file)

    detections = []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            object_name = model.names[class_id]

            detection = {
                "object_name": object_name,
                "confidence": confidence
            }

            detections.append(detection)

            detections_collection.insert_one(detection)

    return {
        "message": "Detection completed",
        "detections": detections
    }
@app.post("/detections")
def create_detection(object_name: str, confidence: float):
    detection = {
        "object_name": object_name,
        "confidence": confidence
    }

    result = detections_collection.insert_one(detection)

    return {
        "message": "Detection saved successfully!",
        "detection_id": str(result.inserted_id)
    }


@app.get("/")
def home():
    return {"message": "Physical World AI Backend is running!"}


@app.post("/users")
def create_user(name: str, email: str):
    user = {
        "name": name,
        "email": email
    }

    result = users_collection.insert_one(user)

    return {
        "message": "User created successfully!",
        "user_id": str(result.inserted_id)
    }
@app.get("/detections")
def get_detections():
    detections = list(detections_collection.find())

    for detection in detections:
        detection["_id"] = str(detection["_id"])

    return detections
