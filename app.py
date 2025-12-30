import streamlit as st
import cv2
import numpy as np
from pathlib import Path

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array


# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Face Mask Detection",
    page_icon="😷",
    layout="centered"
)

st.title("😷 Face Mask Detection (Camera Photo)")


# --------------------------------------------------
# Load models (cached)
# --------------------------------------------------
@st.cache_resource
def load_models():
    face_proto = Path("face_detector/deploy.prototxt")
    face_model = Path("face_detector/res10_300x300_ssd_iter_140000.caffemodel")
    mask_model = Path("mask_detector.model")

    if not face_proto.exists() or not face_model.exists():
        raise FileNotFoundError("Face detector files missing")

    if not mask_model.exists():
        raise FileNotFoundError("mask_detector.model not found")

    face_net = cv2.dnn.readNetFromCaffe(
        str(face_proto),
        str(face_model)
    )

    mask_clf = load_model(str(mask_model))
    return face_net, mask_clf


face_net, mask_clf = load_models()


# --------------------------------------------------
# Sidebar controls
# --------------------------------------------------
st.sidebar.header("Settings")

conf_thresh = st.sidebar.slider(
    "Face detection confidence",
    0.1, 0.95, 0.6, 0.05
)

pad_pct = st.sidebar.slider(
    "Face padding (%)",
    0, 60, 20, 5
) / 100.0

min_face = st.sidebar.slider(
    "Minimum face size (px)",
    20, 200, 70, 5
)


# --------------------------------------------------
# Detection function
# --------------------------------------------------
def detect_and_annotate(image_bgr):
    h, w = image_bgr.shape[:2]

    blob = cv2.dnn.blobFromImage(
        cv2.resize(image_bgr, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0),
    )
    face_net.setInput(blob)
    detections = face_net.forward()

    results = []

    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence < conf_thresh:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype(int)

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        if (x2 - x1) < min_face or (y2 - y1) < min_face:
            continue

        # padding
        bw, bh = (x2 - x1), (y2 - y1)
        x1p = max(0, int(x1 - pad_pct * bw))
        y1p = max(0, int(y1 - pad_pct * bh))
        x2p = min(w - 1, int(x2 + pad_pct * bw))
        y2p = min(h - 1, int(y2 + pad_pct * bh))

        face = image_bgr[y1p:y2p, x1p:x2p]
        if face.size == 0:
            continue

        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_rgb = cv2.resize(face_rgb, (224, 224))

        arr = img_to_array(face_rgb)
        arr = preprocess_input(arr)
        arr = np.expand_dims(arr, axis=0)

        mask, no_mask = mask_clf.predict(arr, verbose=0)[0]
        label = "Mask" if mask > no_mask else "No Mask"
        score = float(max(mask, no_mask))

        color = (0, 200, 0) if label == "Mask" else (0, 0, 255)

        cv2.rectangle(image_bgr, (x1p, y1p), (x2p, y2p), color, 2)
        cv2.putText(
            image_bgr,
            f"{label} ({score:.2f})",
            (x1p, max(0, y1p - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
        )

        results.append(label)

    return image_bgr, results


# --------------------------------------------------
# Camera capture
# --------------------------------------------------
st.subheader("📸 Take a photo")

photo = st.camera_input("Open camera and capture")

if photo is not None:
    file_bytes = np.asarray(bytearray(photo.getvalue()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image_bgr is None:
        st.error("Could not read the image.")
    else:
        output, results = detect_and_annotate(image_bgr)

        st.subheader("Result")
        st.image(output, channels="BGR", use_column_width=True)

        if not results:
            st.info("No face detected. Please face the camera and try again.")
        else:
            masks = results.count("Mask")
            nomasks = results.count("No Mask")
            st.success(f"Detected {len(results)} face(s) —  Mask: {masks} | No Mask: {nomasks}")


# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown(
    """
    <hr>
    <div style="text-align:center; font-size:14px; color:gray;">
        Developed by <b>Hasnat Samiul</b> ✌🏼
    </div>
    """,
    unsafe_allow_html=True,
)
