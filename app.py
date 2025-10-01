import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import cv2
import numpy as np
import mediapipe as mp
import math
import streamlit.components.v1 as components

# Load audio only once
def play_alert():
    components.html(
        """
        <script>
        var sound = new Audio("https://actions.google.com/sounds/v1/alarms/alarm_clock.ogg");
        sound.play();
        </script>
        """,
        height=0,
    )

# Helper function
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

# Streamlit UI
st.set_page_config(page_title="AI Ergonomic Coach", layout="centered")
st.title("🧍 AI Ergonomic Coach for Dental Clinics")
st.write("This app uses your webcam to detect your sitting posture in real time.")

# Pose detection
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

bad_posture_flag = st.empty()

class PostureDetector(VideoTransformerBase):
    def __init__(self):
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.alert_triggered = False

    def transform(self, frame: av.VideoFrame) -> np.ndarray:
        image = frame.to_ndarray(format="bgr24")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            lm = results.pose_landmarks.landmark

            shoulder = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            hip = [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   lm[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                    lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

            angle = calculate_angle(shoulder, hip, knee)

            if angle < 160:
                posture = "⚠️ Poor posture"
                color = (0, 0, 255)
                if not self.alert_triggered:
                    st.session_state['trigger_alert'] = True
                    self.alert_triggered = True
            else:
                posture = "✅ Good posture"
                color = (0, 255, 0)
                self.alert_triggered = False

            cv2.putText(image, f'{int(angle)}°', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(image, posture, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        return image

# Init alert state
if 'trigger_alert' not in st.session_state:
    st.session_state['trigger_alert'] = False

# Launch webcam stream
webrtc_streamer(
    key="posture-coach",
    video_transformer_factory=PostureDetector,
    media_stream_constraints={"video": True, "audio": False},
)

# Sound alert if triggered
if st.session_state['trigger_alert']:
    play_alert()
    bad_posture_flag.warning("⚠️ Poor posture detected! Please sit upright.")
    st.session_state['trigger_alert'] = False


