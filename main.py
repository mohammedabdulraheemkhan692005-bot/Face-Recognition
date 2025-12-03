from fastapi import FastAPI
from pydantic import BaseModel
import base64
import io
from PIL import Image
import numpy as np
import face_recognition
from datetime import datetime

app = FastAPI()

known_encodings = []
known_names = []
attendance_log = []

class ImagePayload(BaseModel):
    image_base64: str
    name: str | None = None

def decode_image(b64_string: str) -> np.ndarray:
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_string)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return np.array(img)


@app.post("/register")
def register_face(payload: ImagePayload):
    img = decode_image(payload.image_base64)
    encodings = face_recognition.face_encodings(img)

    if not encodings:
        return {"status": "error", "message": "No face detected"}

    encoding = encodings[0]
    known_encodings.append(encoding)
    known_names.append(payload.name or f"user_{len(known_names)+1}")

    return {"status": "ok", "message": "Face registered", "name": known_names[-1]}


@app.post("/mark")
def mark_attendance(payload: ImagePayload):
    if not known_encodings:
        return {"status": "error", "message": "No registered faces"}

    img = decode_image(payload.image_base64)
    encodings = face_recognition.face_encodings(img)

    if not encodings:
        return {"status": "error", "message": "No face detected"}

    encoding = encodings[0]
    matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.5)
    
    if True not in matches:
        return {"status": "error", "message": "Unknown face"}

    idx = matches.index(True)
    name = known_names[idx]
    timestamp = datetime.now().isoformat(timespec="seconds")

    attendance_log.append((name, timestamp))

    return {"status": "ok", "name": name, "time": timestamp}


@app.get("/attendance")
def get_attendance():
    return {
        "status": "ok",
        "records": [{"name": n, "time": t} for n, t in attendance_log]
    }
