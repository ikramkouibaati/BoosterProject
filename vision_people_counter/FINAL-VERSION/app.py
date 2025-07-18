# Python In-built packages
from pathlib import Path
import PIL
from datetime import datetime
import os
import csv

# External packages
import streamlit as st

# Local Modules
import settings
import helper

# Setting page layout
st.set_page_config(
    page_title="Object Detection using YOLOv8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Object Detection And Tracking using YOLOv8")

# Sidebar
st.sidebar.header("ML Model Config")

# Model Options
model_type = st.sidebar.radio(
    "Select Task", ['Detection', 'Segmentation'])

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 25, 100, 40)) / 100

# Selecting Detection Or Segmentation
if model_type == 'Detection':
    model_path = Path(settings.DETECTION_MODEL)
elif model_type == 'Segmentation':
    model_path = Path(settings.SEGMENTATION_MODEL)

# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header("Image/Video Config")

# 👤 Seuil maximal défini par l'utilisateur
max_people_allowed = st.sidebar.number_input(
    "Fixer un nombre maximal de personnes",
    min_value=1,
    value=10,
    step=1
)

# 🐶 Option Pet Friendly
pet_friendly = st.sidebar.radio(
    "Salle Pet Friendly 🐾",
    options=["Oui", "Non"],
    index=0
)

source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST)

source_img = None
# If image is selected
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Default Image",
                         use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image",
                         use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            st.image(default_detected_image_path, caption='Detected Image',
                     use_column_width=True)
        else:
            if st.sidebar.button('Detect Objects'):
                res = model.predict(uploaded_image, conf=confidence)
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]

                # Compter les personnes détectées (classe 0)
                person_count = 0
                for box in boxes:
                    cls = int(box.cls[0])
                    if cls == 0:
                        person_count += 1

                # Afficher image + infos
                st.image(res_plotted, caption='Detected Image', use_column_width=True)
                st.info(f"Nombre de personnes détectées : {person_count}")

                if person_count > max_people_allowed:
                    st.warning(f"⚠️ Trop de personnes sur l’image ! (Limite : {max_people_allowed})")
                else:
                    st.success(f"✅ Nombre de personnes dans la limite ({max_people_allowed})")

                # Log dans logs/image_log.csv
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_path = os.path.join("logs", "image_log.csv")
                if not os.path.exists("logs"):
                    os.makedirs("logs")
                with open(log_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([now, person_count])

                try:
                    with st.expander("Detection Results"):
                        for box in boxes:
                            st.write(box.data)
                except Exception as ex:
                    st.write("Erreur lors de l'affichage des résultats.")

elif source_radio == settings.VIDEO:
    # Utilisation d'un nom unique pour cette instance
    helper.play_stored_video(
        confidence,
        model,
        max_people_allowed,
        instance_name="video_simple"  # 🔑 nom unique
    )

elif source_radio == settings.WEBCAM:
    helper.play_webcam(confidence, model, max_people_allowed, instance_name="webcam_main")

elif source_radio == settings.RTSP:
    helper.play_rtsp_stream(confidence, model, max_people_allowed)

elif source_radio == settings.YOUTUBE:
    helper.play_youtube_video(confidence, model, max_people_allowed)

elif source_radio == "Double Video":
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎥 Vidéo 1")
        helper.play_stored_video(
            confidence,
            model,
            max_people_allowed,
            instance_name="video1",  # 🔑 nom unique
            auto_start=True
        )

    with col2:
        st.markdown("### 🎥 Vidéo 2")
        helper.play_stored_video(
            confidence,
            model,
            max_people_allowed,
            instance_name="video2",  # 🔑 nom unique
            auto_start=True
        )

else:
    st.error("Please select a valid source type!")
