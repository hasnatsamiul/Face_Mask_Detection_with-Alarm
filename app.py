import streamlit as st
import cv2
import numpy as np
import time
from pathlib import Path
from base64 import b64encode

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Face Mask Detection (Live)",
    page_icon="😷",
    layout="centered"
)
st.title("😷 Live Face Mask Detection (WebRTC)")

# --------------------------------------------------
# Load models (cached)
# --------------------------------------------------
@st.cache_resource
def load_models():
    face_proto = Path("face_detector/deploy.prototxt")
    face_model = Path("face_detector/res10_300x300_ssd_iter_140000.caffemodel")

    if not face_proto.exists() or not face_model.exists():
        raise FileNotFoundError("Face detector files missing")

    net = cv2.dnn.readNetFromCaffe(
        str(face_proto),
        str(face_model)
    )

    mask_model = Path("mask_detector.model")
    if not mask_model.exists():
        raise FileNotFoundError("mask_detector.model not found")

    clf = load_model(str(mask_model))
    return net, clf


@st.cache_resource
def load_alarm():
    p = Path("alert.mp3")
    if not p.exists():
        return None, None
    data = p.read_bytes()
    return data, b64encode(data).decode()


net, clf = load_models()
alarm_bytes, alarm_b64 = load_alarm()

# --------------------------------------------------
# Session state
# --------------------------------------------------
if "alarm_flag" not in st.session_state:
    st.session_state.alarm_flag = False

if "last_alarm" not in st.session_state:
    st.session_state.last_alarm = 0.0

# --------------------------------------------------
# Detection function
# --------------------------------------------------
def detect_and_annotate(frame, conf_thresh):
    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0),
    )
    net.setInput(blob)
    detections = net.forward()

    has_mask = False
    has_no_mask = False

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < conf_thresh:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype("int")

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            continue

        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_rgb = cv2.resize(face_rgb, (224, 224))

        arr = img_to_array(face_rgb)
        arr = preprocess_input(arr)
        arr = np.expand_dims(arr, axis=0)

        mask, no_mask = clf.predict(arr, verbose=0)[0]
        label = "Mask" if mask > no_mask else "No Mask"
        color = (0, 200, 0) if label == "Mask" else (0, 0, 255)

        if label == "Mask":
            has_mask = True
        else:
            has_no_mask = True

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{label}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    return frame, has_mask, has_no_mask


# --------------------------------------------------
# Sidebar controls
# --------------------------------------------------
st.sidebar.header("Settings")

conf_thresh = st.sidebar.slider(
    "Face detection confidence",
    0.1, 0.9, 0.5, 0.05
)

alarm_enabled = st.sidebar.toggle("Enable alarm", value=True)
alarm_trigger = st.sidebar.selectbox(
    "Alarm when",
    ["No Mask", "Mask"]
)
cooldown = st.sidebar.slider(
    "Alarm cooldown (seconds)",
    0.0, 10.0, 3.0, 0.5
)

if alarm_enabled and alarm_bytes is None:
    st.sidebar.warning("alert.mp3 not found")

# --------------------------------------------------
# WebRTC Video Processor
# --------------------------------------------------
class MaskProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        out, has_mask, has_no_mask = detect_and_annotate(
            img,
            conf_thresh
        )

        trigger = (
            (alarm_trigger == "No Mask" and has_no_mask) or
            (alarm_trigger == "Mask" and has_mask)
        )

        if trigger:
            st.session_state.alarm_flag = True

        return out


# --------------------------------------------------
# Start WebRTC
# --------------------------------------------------
webrtc_streamer(
    key="mask-detection",
    video_processor_factory=MaskProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True,
)

# --------------------------------------------------
# Alarm playback (MAIN THREAD ONLY)
# --------------------------------------------------
now = time.time()
if (
    alarm_enabled
    and st.session_state.alarm_flag
    and alarm_bytes is not None
    and (now - st.session_state.last_alarm) > cooldown
):
    st.warning("🚨 Alarm triggered!")
    st.audio(alarm_bytes, format="audio/mp3")

    st.markdown(
        f"""
        <audio autoplay>
            <source src="data:audio/mp3;base64,{alarm_b64}">
        </audio>
        """,
        unsafe_allow_html=True,
    )

    st.session_state.last_alarm = now
    st.session_state.alarm_flag = False

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown(
    """
    <hr>
    <div style="text-align:center; font-size:14px; color:gray;">
        Developed by <b>Hasnat Samiul</b> ✌🏼<br>
        <a href="mailto:smhasnats@gmail.com">smhasnats@gmail.com</a>
    </div>
    """,
    unsafe_allow_html=True,
)
