from fastapi import FastAPI
from pydantic import BaseModel
import base64
import io
from PIL import Image
import numpy as np
from deepface import DeepFace
from datetime import datetime
import os

# Disable TensorFlow errors for safety
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["DEEPFACE_HOME"] = "/tmp"

app = FastAPI()

# Store registered faces in memory
registered_faces = {}    # {name: numpy array}
attendance_log = []      # [(name, timestamp)]

class ImagePayload(BaseModel):
    image_base64: str
    name: str | None = None


def decode_image(data):
    """Decode base64 → numpy array."""
    if "," in data:
        data = data.split(",")[1]
    img_bytes = base64.b64decode(data)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return np.array(img)


@app.post("/register")
def register_face(payload: ImagePayload):
    img = decode_image(payload.image_base64)
    name = payload.name

    if not name:
        return {"status": "error", "message": "No name provided"}

    registered_faces[name] = img
    return {"status": "ok", "message": f"Face registered for {name}"}


@app.post("/mark")
def mark_attendance(payload: ImagePayload):
    if not registered_faces:
        return {"status": "error", "message": "No faces registered yet"}

    img = decode_image(payload.image_base64)

    # Compare against all registered faces
    for name, reg_img in registered_faces.items():
        try:
            result = DeepFace.verify(
                img,
                reg_img,
                enforce_detection=False,
                model_name="Facenet"        # IMPORTANT: TensorFlow-free model
            )

            if result.get("verified"):
                timestamp = datetime.now().isoformat(timespec="seconds")
                attendance_log.append((name, timestamp))

                return {
                    "status": "ok",
                    "name": name,
                    "time": timestamp
                }
        except Exception as e:
            print("Verification error:", e)

    return {"status": "error", "message": "Face not recognized"}


@app.get("/attendance")
def get_attendance():
    return {
        "status": "ok",
        "records": [
            {"name": n, "time": t}
            for (n, t) in attendance_log
        ]
    }
