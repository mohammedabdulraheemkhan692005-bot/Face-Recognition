from fastapi import FastAPI
from pydantic import BaseModel
import base64, io
from PIL import Image
import numpy as np
from deepface import DeepFace
from datetime import datetime

app = FastAPI()

registered_faces = {}   # {name: numpy array}
attendance_log = []     # list of (name, timestamp)

class ImagePayload(BaseModel):
    image_base64: str
    name: str | None = None

def decode_img(data):
    if "," in data:
        data = data.split(",")[1]
    img_bytes = base64.b64decode(data)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return np.array(img)

@app.post("/register")
def register(payload: ImagePayload):
    img = decode_img(payload.image_base64)
    registered_faces[payload.name] = img
    return {"status": "ok", "message": "Face registered", "name": payload.name}

@app.post("/mark")
def mark(payload: ImagePayload):
    if not registered_faces:
        return {"status": "error", "message": "No registered faces"}

    img = decode_img(payload.image_base64)

    for name, reg_img in registered_faces.items():
        try:
            result = DeepFace.verify(img, reg_img, enforce_detection=False)
            if result["verified"]:
                ts = datetime.now().isoformat(timespec="seconds")
                attendance_log.append((name, ts))
                return {"status": "ok", "name": name, "time": ts}
        except:
            pass

    return {"status": "error", "message": "Unknown face"}

@app.get("/attendance")
def attendance():
    return {
        "status": "ok",
        "records": [{"name": n, "time": t} for (n, t) in attendance_log]
    }
