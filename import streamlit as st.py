import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from posture_utils import calculate_angle
from datetime import datetime

st.set_page_config(page_title="AI Ergonomics Coach", layout="wide")

st.title("🧍‍♂️ AI Ergonomics Coach for Dentistry & Physiotherapy")

run = st.toggle("Start Webcam Tracking")
pose = mp.solutions.pose.Pose()

cap = cv2.VideoCapture(0)

FRAME_WINDOW = st.image([])

angle_display = st.empty()
log = []

while run:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
        hip = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP]
        knee = landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE]

        angle = calculate_angle(shoulder, hip, knee)
        posture = "Good" if 160 < angle < 180 else "Leaning Forward"

        # Annotate frame
        cv2.putText(frame, f"Back Angle: {int(angle)} deg", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Posture: {posture}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        timestamp = datetime.now().strftime('%H:%M:%S')
        log.append((timestamp, int(angle), posture))

    FRAME_WINDOW.image(frame, channels="BGR")

# Save session data
if st.button("📥 Export Session Log"):
    import pandas as pd
    df = pd.DataFrame(log, columns=["Time", "Back Angle", "Posture"])
    df.to_csv("posture_session.csv", index=False)
    st.success("✅ Log saved as posture_session.csv")
