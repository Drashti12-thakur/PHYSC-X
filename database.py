import os
from pymongo import MongoClient
from dotenv import load_dotenv
from urllib.parse import quote_plus

load_dotenv()

username = "tdrishtii06_db_user"
password = os.getenv("MONGO_PASSWORD")

encoded_password = quote_plus(password)

MONGODB_URL = (
    f"mongodb+srv://{username}:{encoded_password}"
    "@cluster0.kep92hu.mongodb.net/?appName=Cluster0"
)

client = MongoClient(MONGODB_URL)

try:
    client.admin.command("ping")
    print("MongoDB connected successfully!")
except Exception as e:
    print("MongoDB connection failed:", e)
