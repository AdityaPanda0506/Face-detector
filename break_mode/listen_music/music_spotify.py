import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import webbrowser
import os

# Load model and labels
try:
    model = load_model("model.h5")
    label = np.load("labels.npy")
except Exception as e:
    st.error(f"Error loading model or labels: {e}")
    st.stop()

# Mediapipe setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

holis = mp_holistic.Holistic()
drawing = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

st.header("Emotion Based Spotify Recommender")

# Session state initialization
if "run" not in st.session_state:
    st.session_state["run"] = True

try:
    emotion = np.load("emotion.npy")[0]
except:
    emotion = ""

if not emotion:
    st.session_state["run"] = True
else:
    st.session_state["run"] = False

# Video Frame Processor Class
class EmotionProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        frm = frame.to_ndarray(format="bgr24")
        frm = cv2.flip(frm, 1)
        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

        lst = []
        if res.face_landmarks:
            for i in res.face_landmarks.landmark:
                lst.append(i.x - res.face_landmarks.landmark[1].x)
                lst.append(i.y - res.face_landmarks.landmark[1].y)

            # Left hand landmarks
            if res.left_hand_landmarks:
                for i in res.left_hand_landmarks.landmark:
                    lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
            else:
                lst.extend([0.0] * 42)

            # Right hand landmarks
            if res.right_hand_landmarks:
                for i in res.right_hand_landmarks.landmark:
                    lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
            else:
                lst.extend([0.0] * 42)

            lst = np.array(lst).reshape(1, -1)
            try:
                pred = label[np.argmax(model.predict(lst))]
                cv2.putText(frm, pred, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                np.save("emotion.npy", np.array([pred]))
            except Exception as e:
                st.warning("Prediction error: " + str(e))

        # Draw landmarks
        mp_drawing.draw_landmarks(frm, res.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                                  landmark_drawing_spec=drawing,
                                  connection_drawing_spec=drawing)
        mp_drawing.draw_landmarks(frm, res.left_hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(frm, res.right_hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

        return av.VideoFrame.from_ndarray(frm, format="bgr24")


# UI inputs
lang = st.text_input("Language")
singer = st.text_input("Preferred Singer")

if lang and singer and st.session_state["run"]:
    webrtc_streamer(
        key="emotion-stream",
        video_processor_factory=EmotionProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    )

btn = st.button("Recommend me songs")

if btn:
    if not emotion:
        st.warning("Please let me capture your emotion first.")
        st.session_state["run"] = True
    else:
        search_query = f"{lang} {emotion} song {singer}".replace(" ", "+")
        spotify_url = f"https://open.spotify.com/search/ "

        st.markdown(f"""
        🎵 Opening Spotify to search for:  
        **{lang} {emotion} song by {singer}**

        ⚠️ Note: Spotify does not support direct search from URLs.  
        Please search manually in the opened window.
        """)

        webbrowser.open(spotify_url)

        # Reset emotion
        np.save("emotion.npy", np.array([""]))
        st.session_state["run"] = False