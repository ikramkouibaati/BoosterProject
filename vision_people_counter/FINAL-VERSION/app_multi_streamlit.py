
import streamlit as st
import cv2
from ultralytics import YOLO
from datetime import datetime
import os
import csv
import tempfile

# === CONFIG ===
SOURCES = {
    "Webcam": 0,
    "Vidéo 1": "videos/video1.mp4",
    "Vidéo 2": "videos/video2.mp4"
}
MODEL_PATH = "weights/yolov8n.pt"
LOG_DIR = "logs"
CONFIDENCE = 0.3

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

model = YOLO(MODEL_PATH)
_last_logged_count = {}

def log_people_count_if_changed(count, source_name):
    global _last_logged_count
    log_path = os.path.join(LOG_DIR, f"{source_name}_log.csv")

    if _last_logged_count.get(source_name) == count:
        return
    _last_logged_count[source_name] = count

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([now, count])

def detect_and_display(source_label, source):
    st.subheader(f"🎥 Flux : {source_label}")
    frame_holder = st.empty()
    cap = cv2.VideoCapture(source)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model.predict(frame, conf=CONFIDENCE)
        boxes = results[0].boxes
        count = sum(1 for box in boxes if int(box.cls[0]) == 0)

        log_people_count_if_changed(count, source_label)

        annotated = results[0].plot()
        frame_holder.image(annotated, channels="BGR", caption=f"{count} personne(s) détectée(s)")

        if st.button(f"⛔ Arrêter {source_label}", key=source_label):
            break

    cap.release()

st.set_page_config(layout="wide")
st.title("🎯 Surveillance Multi-Flux avec Logs Séparés")

tab_objects = st.tabs(list(SOURCES.keys()))
for tab, (label, src) in zip(tab_objects, SOURCES.items()):
    with tab:
        detect_and_display(label, src)
