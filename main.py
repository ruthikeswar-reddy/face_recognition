

# import os
# import io
# import re
# import pickle
# import numpy as np
# import cv2
# import streamlit as st
# import face_recognition

# # --- Setup ---
# st.set_page_config(page_title="Face Registration & Recognition", layout="wide")

# # Directories
# EMBEDDING_DIR = "./face_embeddings"
# os.makedirs(EMBEDDING_DIR, exist_ok=True)
# embeddings = dict()

# # Load Haar cascade for eye detection
# eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# # --- Utility Functions ---
# def eyes_are_open(image_np: np.ndarray) -> bool:
#     gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
#     eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
#     return len(eyes) >= 2


# def save_embedding(name: str, embedding: np.ndarray):
#     sanitized = re.sub(r'[^\w]', '_', name.strip())
#     path = os.path.join(EMBEDDING_DIR, f"{sanitized}.pickle")
#     with open(path, "wb") as f:
#         pickle.dump(embedding, f)


# def load_embeddings():
#     embeddings = {}
#     for fname in os.listdir(EMBEDDING_DIR):
#         if fname.endswith(".pickle"):
#             name = os.path.splitext(fname)[0]
#             with open(os.path.join(EMBEDDING_DIR, fname), "rb") as f:
#                 embeddings[name] = pickle.load(f)
#     return embeddings

# # --- Streamlit UI ---
# st.title("🧑‍💻 Face Registration & Recognition via Webcam")
# mode = st.sidebar.selectbox("Choose Mode", ["Register", "Recognize"])

# if mode == "Register":
#     st.header("Register a New User")
#     username = st.text_input("Enter Username")
#     camera_input = st.camera_input("Capture Face Image (ensure eyes are open)")
#     if st.button("Register"):
#         if not username:
#             st.error("Please enter a username.")
#         elif not camera_input:
#             st.error("Please capture an image using your webcam.")
#         else:
#             # Read image bytes
#             image_bytes = camera_input.read()
#             file_bytes = np.frombuffer(image_bytes, dtype=np.uint8)
#             img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
#             img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

#             if not eyes_are_open(img_rgb):
#                 st.error("Eyes are not open. Please try again.")
#             else:
#                 encodings = face_recognition.face_encodings(img_rgb)
#                 if not encodings:
#                     st.error("No face detected. Please try another capture.")
#                 else:
#                     save_embedding(username, encodings[0])
#                     embeddings[username] = encodings[0]
#                     st.success(f"User '{username}' registered successfully.")

# elif mode == "Recognize":
#     st.header("Live Recognition Stream")
#     if 'run' not in st.session_state:
#         st.session_state.run = False
#     if st.button("Start Recognition"):
#         st.session_state.run = True
#     if st.button("Stop Recognition"):
#         st.session_state.run = False

#     placeholder = st.empty()
#     known = load_embeddings()
#     if st.session_state.run:
#         cap = cv2.VideoCapture(0)
#         while st.session_state.run:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             # Process frame
#             img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             name = "unknown"
#             if eyes_are_open(img_rgb):
#                 encs = face_recognition.face_encodings(img_rgb)
#                 if encs:
#                     for person, emb in known.items():
#                         if face_recognition.compare_faces([emb], encs[0])[0]:
#                             name = person
#                             break
#             # Overlay name at top-right
#             h, w, _ = frame.shape
#             text = f"{name}"
#             font = cv2.FONT_HERSHEY_SIMPLEX
#             scale = 1
#             thickness = 2
#             (txt_w, txt_h), _ = cv2.getTextSize(text, font, scale, thickness)
#             x = w - txt_w - 10
#             y = txt_h + 10
#             cv2.putText(frame, text, (x, y), font, scale, (0, 255, 0), thickness)
#             # Display
#             placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
#         cap.release()

# # --- Footer ---
# st.markdown("---")
# st.write("Developed with ❤️ using Streamlit and Face Recognition.")


import os
import re
import pickle
import numpy as np
import cv2
import streamlit as st
import face_recognition
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# --- Setup ---
st.set_page_config(page_title="Face Registration & Recognition", layout="wide")

# Directories
EMBEDDING_DIR = "./face_embeddings"
os.makedirs(EMBEDDING_DIR, exist_ok=True)

# Load Haar cascade for eye detection
eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# --- Utility Functions ---
def eyes_are_open(image_np: np.ndarray) -> bool:
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    eyes = eyes_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return len(eyes) >= 2


def save_embedding(name: str, embedding: np.ndarray):
    sanitized = re.sub(r'[^\w]', '_', name.strip())
    path = os.path.join(EMBEDDING_DIR, f"{sanitized}.pickle")
    with open(path, "wb") as f:
        pickle.dump(embedding, f)


def load_embeddings():
    embeddings = {}
    for fname in os.listdir(EMBEDDING_DIR):
        if fname.endswith(".pickle"):
            name = os.path.splitext(fname)[0]
            with open(os.path.join(EMBEDDING_DIR, fname), "rb") as f:
                embeddings[name] = pickle.load(f)
    return embeddings


# --- Streamlit UI ---
st.title("🧑‍💻 Face Registration & Recognition via Webcam")
mode = st.sidebar.selectbox("Choose Mode", ["Register", "Recognize"])

if mode == "Register":
    st.header("Register a New User")
    username = st.text_input("Enter Username")
    camera_input = st.camera_input("Capture Face Image (ensure eyes are open)")
    if st.button("Register"):
        if not username:
            st.error("Please enter a username.")
        elif not camera_input:
            st.error("Please capture an image using your webcam.")
        else:
            # Read image bytes
            image_bytes = camera_input.read()
            file_bytes = np.frombuffer(image_bytes, dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            if not eyes_are_open(img_rgb):
                st.error("Eyes are not open. Please try again.")
            else:
                encodings = face_recognition.face_encodings(img_rgb)
                if not encodings:
                    st.error("No face detected. Please try another capture.")
                else:
                    save_embedding(username, encodings[0])
                    st.success(f"User '{username}' registered successfully.")

elif mode == "Recognize":
    st.header("Live Recognition Stream")
    known = load_embeddings()

    class VideoProcessor(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            name = "unknown"
            if eyes_are_open(img_rgb):
                encs = face_recognition.face_encodings(img_rgb)
                if encs:
                    for person, emb in known.items():
                        if face_recognition.compare_faces([emb], encs[0])[0]:
                            name = person
                            break
            # Overlay name at top-right
            h, w, _ = img.shape
            (txt_w, txt_h), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            x = w - txt_w - 10
            y = txt_h + 10
            cv2.putText(img, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            return img

    # Start the WebRTC streamer
    webrtc_streamer(key="recognition", video_processor_factory=VideoProcessor)

# --- Footer ---
st.markdown("---")
st.write("Developed with ❤️Ruthik using Streamlit, Face Recognition, and WebRTC.")
