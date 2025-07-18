from ultralytics import YOLO
import streamlit as st
import cv2
import yt_dlp
import settings
import csv
import os
from datetime import datetime

_last_logged_count = {}  # dictionnaire global pour suivre le dernier nombre de personnes par source

def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model


def display_tracker_options(key_suffix=""):
    display_tracker = st.radio("Display Tracker", ('Yes', 'No'), key=f"display_tracker_{key_suffix}")
    is_display_tracker = True if display_tracker == 'Yes' else False
    if is_display_tracker:
        tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"), key=f"tracker_type_{key_suffix}")
        return is_display_tracker, tracker_type
    return is_display_tracker, None


def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=None, tracker=None, source_name="default", max_people=None):
    """
    Display the detected objects on a video frame using the YOLOv8 model.

    Args:
    - conf (float): Confidence threshold for object detection.
    - model (YoloV8): A YOLOv8 object detection model.
    - st_frame (Streamlit object): A Streamlit object to display the detected video.
    - image (numpy array): A numpy array representing the video frame.
    - is_display_tracking (bool): A flag indicating whether to display object tracking (default=None).
    - source_name (str): Name of the video source (used for logging purposes).

    Returns:
    None
    """

    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720*(9/16))))

    # Detect or track objects
    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True, tracker=tracker)
    else:
        res = model.predict(image, conf=conf)

    # Count number of persons (class 0 in COCO)
    boxes = res[0].boxes
    person_count = 0
    for box in boxes:
        cls = int(box.cls[0])
        if cls == 0:  # class 0 = person
            person_count += 1

    # Call logging function only if person count changed
    log_people_count_if_changed(person_count, source_name)

    # 👇 Affichage du warning en cas de dépassement
    if max_people is not None and person_count > max_people:
        st.warning(f"⚠️ Nombre de personnes détectées ({person_count}) dépasse la limite autorisée ({max_people}) !")

    # Display result
    res_plotted = res[0].plot()
    st_frame.image(
        res_plotted,
        caption='Detected Video',
        channels="BGR",
        use_column_width=True
    )

def get_youtube_stream_url(youtube_url):
    ydl_opts = {
        'format': 'best[ext=mp4]',
        'no_warnings': True,
        'quiet': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        return info['url']


def play_youtube_video(conf, model, max_people_allowed):    
    source_youtube = st.sidebar.text_input("YouTube Video url")
    is_display_tracker, tracker = display_tracker_options()

    

    if st.sidebar.button('Detect Objects'):
        if not source_youtube:
            st.sidebar.error("Please enter a YouTube URL")
            return

        try:
            st.sidebar.info("Extracting video stream URL...")
            stream_url = get_youtube_stream_url(source_youtube)

            st.sidebar.info("Opening video stream...")
            vid_cap = cv2.VideoCapture(stream_url)

            if not vid_cap.isOpened():
                st.sidebar.error("Failed to open video stream. Please try a different video.")
                return

            st.sidebar.success("Video stream opened successfully!")
            st_frame = st.empty()
            source_name = "youtube"  # 👈 utilisé dans les logs
            while vid_cap.isOpened():
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(
                        conf,
                        model,
                        st_frame,
                        image,
                        is_display_tracker,
                        tracker,
                        source_name ,
                        max_people= max_people_allowed
                         
                    )
                else:
                    break

            vid_cap.release()

        except Exception as e:
            st.sidebar.error(f"An error occurred: {str(e)}")



def play_rtsp_stream(conf, model):
   
    source_rtsp = st.sidebar.text_input("rtsp stream url:")
    st.sidebar.caption(
        'Example URL: rtsp://admin:12345@192.168.1.210:554/Streaming/Channels/101')
    is_display_tracker, tracker = display_tracker_options()
    if st.sidebar.button('Detect Objects'):
        try:
            vid_cap = cv2.VideoCapture(source_rtsp)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            vid_cap.release()
            st.sidebar.error("Error loading RTSP stream: " + str(e))


def play_webcam(conf, model, max_people_allowed, instance_name="webcam", auto_start=False):

    import pandas as pd

    source_webcam = settings.WEBCAM_PATH
    is_display_tracker, tracker = display_tracker_options(key_suffix=instance_name)

    run_key = f"run_{instance_name}"
    if run_key not in st.session_state:
        st.session_state[run_key] = auto_start

    # ▶️ ⏹️ Start/Stop boutons
    col1, col2 = st.sidebar.columns(2)
    if col1.button("▶️ Start Webcam", key=f"start_{instance_name}"):
        st.session_state[run_key] = True
    if col2.button("⏹️ Stop Webcam", key=f"stop_{instance_name}"):
        st.session_state[run_key] = False

    st_frame = st.empty()
    source_name = instance_name  # utilisé pour logs et dashboard

    if st.session_state[run_key]:
        try:
            vid_cap = cv2.VideoCapture(source_webcam)

            while vid_cap.isOpened() and st.session_state[run_key]:
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(
                        conf,
                        model,
                        st_frame,
                        image,
                        is_display_tracker,
                        tracker,
                        source_name,
                        max_people=max_people_allowed
                    )
                else:
                    break

            vid_cap.release()

        except Exception as e:
            st.sidebar.error("Error loading webcam: " + str(e))

    # 📊 Dashboard après arrêt
    if not st.session_state[run_key]:
        with st.expander(f"📊 Dashboard - {instance_name}"):
            log_path = os.path.join("logs", f"{instance_name}_log.csv")
            if os.path.exists(log_path):
                df = pd.read_csv(log_path, names=["timestamp", "person_count"])
                if not df.empty:
                    latest_count = int(df["person_count"].iloc[-1])
                    st.metric("👥 Dernier comptage", value=latest_count)
                    st.line_chart(df.tail(20).set_index("timestamp"))
                else:
                    st.info("Aucune donnée dans le log.")
            else:
                st.info("Aucun fichier log trouvé pour cette vidéo.")


def play_stored_video(conf, model, max_people_allowed, instance_name="video", auto_start=False):
    import pandas as pd

    run_key = f"run_{instance_name}"
    if run_key not in st.session_state:
        st.session_state[run_key] = auto_start

    source_vid = st.sidebar.selectbox(
        "Choose a video...",
        settings.VIDEOS_DICT.keys(),
        key=f"video_select_{instance_name}"
    )

    is_display_tracker, tracker = display_tracker_options(key_suffix=instance_name)

    video_path = settings.VIDEOS_DICT.get(source_vid)
    if not video_path:
        st.sidebar.error("Selected video not found in VIDEOS_DICT.")
        return

    try:
        with open(video_path, 'rb') as video_file:
            video_bytes = video_file.read()
        if video_bytes:
            st.video(video_bytes)
    except Exception as e:
        st.sidebar.error(f"Error reading video file: {str(e)}")
        return

    col1, col2 = st.sidebar.columns(2)
    if col1.button("▶️ Run", key=f"run_button_{instance_name}"):
        st.session_state[run_key] = True
    if col2.button("⏹️ Stop", key=f"stop_button_{instance_name}"):
        st.session_state[run_key] = False

    if st.session_state[run_key]:
        try:
            vid_cap = cv2.VideoCapture(str(video_path))
            st_frame = st.empty()
            source_name = instance_name

            while vid_cap.isOpened() and st.session_state[run_key]:
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(
                        conf,
                        model,
                        st_frame,
                        image,
                        is_display_tracker,
                        tracker,
                        source_name,
                        max_people=max_people_allowed
                    )
                else:
                    break

            vid_cap.release()
        except Exception as e:
            st.sidebar.error("Error during detection: " + str(e))

    if not st.session_state[run_key]:
        with st.expander(f"📊 Dashboard - {instance_name}"):
            log_path = os.path.join("logs", f"{instance_name}_log.csv")
            if os.path.exists(log_path):
                df = pd.read_csv(log_path, names=["timestamp", "person_count"])
                if not df.empty:
                    latest_count = int(df["person_count"].iloc[-1])
                    st.metric("👥 Dernier comptage", value=latest_count)
                    st.line_chart(df.tail(20).set_index("timestamp"))
                else:
                    st.info("Aucune donnée dans le log.")
            else:
                st.info("Aucun fichier log trouvé pour cette vidéo.")


_last_logged_count = {}  # global

def log_people_count_if_changed(count, source_name="default", log_dir="logs"):
    """
    Log le nombre de personnes détectées s’il a changé (pas de doublons).
    Écrit dans logs/<source>_log.csv et affiche dans Streamlit.
    """
    global _last_logged_count

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    previous_count = _last_logged_count.get(source_name)
    if previous_count is not None and previous_count == count:
        return  # aucun changement

    _last_logged_count[source_name] = count

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_path = os.path.join(log_dir, f"{source_name}_log.csv")

    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([now, count])

    st.info(f"[{now}] Nombre de personnes : {count}")
