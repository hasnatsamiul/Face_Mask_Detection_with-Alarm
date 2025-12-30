import streamlit as st
import cv2
import numpy as np
import time
from pathlib import Path
from base64 import b64encode
import av

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
    layout="centered",
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
        raise FileNotFoundError("Face detector files missing (deploy.prototxt / .caffemodel)")

    net = cv2.dnn.readNetFromCaffe(str(face_proto), str(face_model))

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
# Sidebar controls
# --------------------------------------------------
st.sidebar.header("Settings")

conf_thresh = st.sidebar.slider("Face detection confidence", 0.1, 0.95, 0.70, 0.05)

pad_pct = st.sidebar.slider("Face padding (%)", 0, 60, 20, 5) / 100.0

min_face = st.sidebar.slider("Min face size (px)", 20, 200, 70, 5)

lighting_boost = st.sidebar.toggle("Lighting boost", value=True)

smooth_alpha = st.sidebar.slider("Smoothing (alpha)", 0.0, 0.95, 0.80, 0.05)
st.sidebar.caption("Higher alpha = more stable label, slower to change.")

alarm_enabled = st.sidebar.toggle("Enable alarm", value=True)
alarm_trigger = st.sidebar.selectbox("Alarm when", ["No Mask", "Mask"])
cooldown = st.sidebar.slider("Alarm cooldown (seconds)", 0.0, 10.0, 3.0, 0.5)

if alarm_enabled and alarm_bytes is None:
    st.sidebar.warning("alert.mp3 not found")


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def boost_contrast_bgr(img_bgr: np.ndarray) -> np.ndarray:
    # Histogram equalization on luminance channel (helps indoor webcam lighting)
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y = cv2.equalizeHist(y)
    merged = cv2.merge([y, cr, cb])
    return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)


def detect_and_annotate(frame_bgr: np.ndarray, conf: float, pad: float, min_face_px: int, alpha: float, state):
    """
    state: dict-like object to store smoothing between frames
    returns: (output_frame, has_mask, has_no_mask)
    """
    h, w = frame_bgr.shape[:2]

    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame_bgr, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0),
    )
    net.setInput(blob)
    detections = net.forward()

    has_mask = False
    has_no_mask = False

    # default smoothing state
    if "p_mask_smooth" not in state:
        state["p_mask_smooth"] = 0.5

    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence < conf:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype("int")

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        # skip tiny detections
        if (x2 - x1) < min_face_px or (y2 - y1) < min_face_px:
            continue

        # padding around face box
        bw = x2 - x1
        bh = y2 - y1
        x1p = max(0, int(x1 - pad * bw))
        y1p = max(0, int(y1 - pad * bh))
        x2p = min(w - 1, int(x2 + pad * bw))
        y2p = min(h - 1, int(y2 + pad * bh))

        face = frame_bgr[y1p:y2p, x1p:x2p]
        if face.size == 0:
            continue

        # preprocess
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_rgb = cv2.resize(face_rgb, (224, 224))

        arr = img_to_array(face_rgb)
        arr = preprocess_input(arr)
        arr = np.expand_dims(arr, axis=0)

        mask, no_mask = clf.predict(arr, verbose=0)[0]
        p_mask = float(mask)

        # smoothing to reduce flicker
        state["p_mask_smooth"] = alpha * state["p_mask_smooth"] + (1.0 - alpha) * p_mask
        p = state["p_mask_smooth"]

        label = "Mask" if p >= 0.5 else "No Mask"
        color = (0, 200, 0) if label == "Mask" else (0, 0, 255)

        if label == "Mask":
            has_mask = True
        else:
            has_no_mask = True

        cv2.rectangle(frame_bgr, (x1p, y1p), (x2p, y2p), color, 2)
        cv2.putText(
            frame_bgr,
            f"{label} ({p:.2f})",
            (x1p, max(0, y1p - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    return frame_bgr, has_mask, has_no_mask


# --------------------------------------------------
# WebRTC Video Processor
# --------------------------------------------------
class MaskProcessor(VideoProcessorBase):
    def __init__(self):
        self.state = {}
        self.triggered = False  # read by main thread

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        if lighting_boost:
            img = boost_contrast_bgr(img)

        out, has_mask, has_no_mask = detect_and_annotate(
            img,
            conf=conf_thresh,
            pad=pad_pct,
            min_face_px=min_face,
            alpha=smooth_alpha,
            state=self.state,
        )

        trigger = (
            (alarm_trigger == "No Mask" and has_no_mask) or
            (alarm_trigger == "Mask" and has_mask)
        )
        if trigger:
            self.triggered = True

        return av.VideoFrame.from_ndarray(out, format="bgr24")


# --------------------------------------------------
# Start WebRTC
# --------------------------------------------------
webrtc_ctx = webrtc_streamer(
    key="mask-detection",
    video_processor_factory=MaskProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# --------------------------------------------------
# Alarm playback (MAIN THREAD)
# --------------------------------------------------
if "last_alarm" not in st.session_state:
    st.session_state.last_alarm = 0.0

now = time.time()

if webrtc_ctx.video_processor:
    # read & reset trigger flag safely in main thread
    vp = webrtc_ctx.video_processor
    triggered = getattr(vp, "triggered", False)
    if triggered:
        vp.triggered = False

    if (
        alarm_enabled
        and triggered
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
