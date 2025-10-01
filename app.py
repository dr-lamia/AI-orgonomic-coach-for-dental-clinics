import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode, RTCConfiguration
import av
import cv2
import numpy as np
import mediapipe as mp
import math
import streamlit.components.v1 as components
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import io
import base64

# Configure STUN/TURN servers for WebRTC
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
    ]}
)

# Initialize session state
def init_session_state():
    if 'trigger_alert' not in st.session_state:
        st.session_state['trigger_alert'] = False
    if 'posture_data' not in st.session_state:
        st.session_state['posture_data'] = []
    if 'session_start' not in st.session_state:
        st.session_state['session_start'] = datetime.now()
    if 'poor_posture_count' not in st.session_state:
        st.session_state['poor_posture_count'] = 0
    if 'good_posture_count' not in st.session_state:
        st.session_state['good_posture_count'] = 0
    if 'total_frames' not in st.session_state:
        st.session_state['total_frames'] = 0
    if 'angle_history' not in st.session_state:
        st.session_state['angle_history'] = []

# Load audio alert
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

# Generate PDF Report
def generate_pdf_report():
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    
    # Container for the 'Flowable' objects
    elements = []
    
    # Define styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=12,
        spaceBefore=12
    )
    
    # Title
    title = Paragraph("🧍 Ergonomic Posture Analysis Report", title_style)
    elements.append(title)
    elements.append(Spacer(1, 12))
    
    # Report metadata
    session_duration = datetime.now() - st.session_state['session_start']
    duration_minutes = int(session_duration.total_seconds() / 60)
    
    metadata = [
        ['Report Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        ['Session Duration:', f'{duration_minutes} minutes'],
        ['Total Frames Analyzed:', str(st.session_state['total_frames'])],
    ]
    
    t = Table(metadata, colWidths=[2.5*inch, 3.5*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ecf0f1')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey)
    ]))
    elements.append(t)
    elements.append(Spacer(1, 20))
    
    # Summary Statistics
    elements.append(Paragraph("Summary Statistics", heading_style))
    
    total_posture = st.session_state['poor_posture_count'] + st.session_state['good_posture_count']
    poor_percentage = (st.session_state['poor_posture_count'] / total_posture * 100) if total_posture > 0 else 0
    good_percentage = (st.session_state['good_posture_count'] / total_posture * 100) if total_posture > 0 else 0
    
    avg_angle = np.mean(st.session_state['angle_history']) if st.session_state['angle_history'] else 0
    min_angle = np.min(st.session_state['angle_history']) if st.session_state['angle_history'] else 0
    max_angle = np.max(st.session_state['angle_history']) if st.session_state['angle_history'] else 0
    
    stats_data = [
        ['Metric', 'Value'],
        ['Good Posture Detections', f"{st.session_state['good_posture_count']} ({good_percentage:.1f}%)"],
        ['Poor Posture Detections', f"{st.session_state['poor_posture_count']} ({poor_percentage:.1f}%)"],
        ['Average Angle', f"{avg_angle:.1f}°"],
        ['Minimum Angle', f"{min_angle:.1f}°"],
        ['Maximum Angle', f"{max_angle:.1f}°"],
    ]
    
    stats_table = Table(stats_data, colWidths=[3*inch, 3*inch])
    stats_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(stats_table)
    elements.append(Spacer(1, 20))
    
    # Recommendations
    elements.append(Paragraph("Recommendations", heading_style))
    
    if poor_percentage > 50:
        recommendation_text = """
        <b>⚠️ Action Required:</b> Your posture analysis shows that you spent more than 50% of the time 
        in poor posture. Consider the following recommendations:
        <br/><br/>
        • Take regular breaks every 30 minutes<br/>
        • Adjust your chair height and screen position<br/>
        • Perform stretching exercises between patients<br/>
        • Consider ergonomic equipment upgrades<br/>
        • Schedule a professional ergonomic assessment
        """
    elif poor_percentage > 30:
        recommendation_text = """
        <b>⚡ Improvement Needed:</b> Your posture shows room for improvement. Consider:
        <br/><br/>
        • Increase awareness of your sitting position<br/>
        • Set reminders to check posture every 20 minutes<br/>
        • Strengthen core muscles with targeted exercises<br/>
        • Review your workstation setup
        """
    else:
        recommendation_text = """
        <b>✅ Good Work!</b> Your posture maintenance is excellent. Keep it up with:
        <br/><br/>
        • Continue your current ergonomic practices<br/>
        • Maintain regular breaks and stretching<br/>
        • Stay mindful of posture during long procedures<br/>
        • Share your best practices with colleagues
        """
    
    elements.append(Paragraph(recommendation_text, styles['Normal']))
    elements.append(Spacer(1, 20))
    
    # Footer
    footer_text = Paragraph(
        "<i>Generated by AI Ergonomic Coach for Dental Clinics</i>",
        ParagraphStyle('Footer', parent=styles['Normal'], fontSize=9, textColor=colors.grey, alignment=TA_CENTER)
    )
    elements.append(Spacer(1, 30))
    elements.append(footer_text)
    
    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer

# Posture Detector Class
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

class PostureDetector(VideoTransformerBase):
    def __init__(self):
        self.pose = mp_pose.Pose(
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5,
            model_complexity=1
        )
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
            
            # Store angle history
            st.session_state['angle_history'].append(angle)
            if len(st.session_state['angle_history']) > 1000:
                st.session_state['angle_history'].pop(0)
            
            st.session_state['total_frames'] += 1

            if angle < 160:
                posture = "⚠️ Poor posture"
                color = (0, 0, 255)
                st.session_state['poor_posture_count'] += 1
                if not self.alert_triggered:
                    st.session_state['trigger_alert'] = True
                    self.alert_triggered = True
                    
                    # Log posture event
                    st.session_state['posture_data'].append({
                        'timestamp': datetime.now(),
                        'angle': angle,
                        'status': 'poor'
                    })
            else:
                posture = "✅ Good posture"
                color = (0, 255, 0)
                st.session_state['good_posture_count'] += 1
                self.alert_triggered = False
                
                # Log posture event
                st.session_state['posture_data'].append({
                    'timestamp': datetime.now(),
                    'angle': angle,
                    'status': 'good'
                })

            cv2.putText(image, f'{int(angle)}°', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(image, posture, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        return image

    def __del__(self):
        if hasattr(self, 'pose'):
            self.pose.close()

# Main App
st.set_page_config(page_title="AI Ergonomic Coach", layout="wide", page_icon="🧍")

init_session_state()

# Sidebar navigation
page = st.sidebar.radio("Navigation", ["🎥 Live Monitoring", "📊 Dashboard", "📄 Reports"])

if page == "🎥 Live Monitoring":
    st.title("🧍 AI Ergonomic Coach for Dental Clinics")
    st.write("This app uses your webcam to detect your sitting posture in real time.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        bad_posture_flag = st.empty()
        
        # Launch webcam stream
        webrtc_ctx = webrtc_streamer(
            key="posture-coach",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_transformer_factory=PostureDetector,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        
        # Sound alert if triggered
        if st.session_state['trigger_alert']:
            play_alert()
            bad_posture_flag.warning("⚠️ Poor posture detected! Please sit upright.")
            st.session_state['trigger_alert'] = False
    
    with col2:
        st.subheader("Session Stats")
        session_duration = datetime.now() - st.session_state['session_start']
        st.metric("Session Duration", f"{int(session_duration.total_seconds() / 60)} min")
        st.metric("Total Frames", st.session_state['total_frames'])
        st.metric("Good Posture", st.session_state['good_posture_count'])
        st.metric("Poor Posture", st.session_state['poor_posture_count'])
        
        if st.button("🔄 Reset Session"):
            for key in ['posture_data', 'poor_posture_count', 'good_posture_count', 'total_frames', 'angle_history']:
                st.session_state[key] = [] if key in ['posture_data', 'angle_history'] else 0
            st.session_state['session_start'] = datetime.now()
            st.rerun()

elif page == "📊 Dashboard":
    st.title("📊 Posture Analysis Dashboard")
    
    if len(st.session_state['posture_data']) == 0:
        st.info("No data available yet. Start monitoring to see analytics.")
    else:
        # Convert to DataFrame
        df = pd.DataFrame(st.session_state['posture_data'])
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_events = len(df)
        poor_count = len(df[df['status'] == 'poor'])
        good_count = len(df[df['status'] == 'good'])
        avg_angle = df['angle'].mean()
        
        col1.metric("Total Events", total_events)
        col2.metric("Good Posture", f"{good_count} ({good_count/total_events*100:.1f}%)")
        col3.metric("Poor Posture", f"{poor_count} ({poor_count/total_events*100:.1f}%)")
        col4.metric("Avg Angle", f"{avg_angle:.1f}°")
        
        # Charts
        st.subheader("Posture Distribution")
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart
            posture_counts = df['status'].value_counts()
            fig_pie = px.pie(
                values=posture_counts.values, 
                names=['Good Posture' if x == 'good' else 'Poor Posture' for x in posture_counts.index],
                color_discrete_sequence=['#2ecc71', '#e74c3c']
            )
            fig_pie.update_layout(title="Posture Status Distribution")
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Angle distribution
            fig_hist = px.histogram(df, x='angle', nbins=30, title="Angle Distribution")
            fig_hist.add_vline(x=160, line_dash="dash", line_color="red", annotation_text="Threshold (160°)")
            st.plotly_chart(fig_hist, use_container_width=True)
        
        # Time series
        st.subheader("Angle Over Time")
        if len(st.session_state['angle_history']) > 0:
            angle_df = pd.DataFrame({
                'Frame': range(len(st.session_state['angle_history'])),
                'Angle': st.session_state['angle_history']
            })
            fig_time = px.line(angle_df, x='Frame', y='Angle', title='Posture Angle Timeline')
            fig_time.add_hline(y=160, line_dash="dash", line_color="red", annotation_text="Good Posture Threshold")
            st.plotly_chart(fig_time, use_container_width=True)
        
        # Recent events table
        st.subheader("Recent Posture Events")
        recent_df = df.tail(20).copy()
        recent_df['timestamp'] = recent_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        recent_df['angle'] = recent_df['angle'].round(1)
        st.dataframe(recent_df, use_container_width=True)

elif page == "📄 Reports":
    st.title("📄 Generate PDF Report")
    
    st.write("Generate a comprehensive PDF report of your posture analysis session.")
    
    if len(st.session_state['posture_data']) == 0:
        st.warning("No data available. Start monitoring to generate reports.")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Report Preview")
            
            session_duration = datetime.now() - st.session_state['session_start']
            duration_minutes = int(session_duration.total_seconds() / 60)
            
            total_posture = st.session_state['poor_posture_count'] + st.session_state['good_posture_count']
            poor_percentage = (st.session_state['poor_posture_count'] / total_posture * 100) if total_posture > 0 else 0
            
            st.write(f"**Session Duration:** {duration_minutes} minutes")
            st.write(f"**Total Frames Analyzed:** {st.session_state['total_frames']}")
            st.write(f"**Good Posture:** {st.session_state['good_posture_count']} frames")
            st.write(f"**Poor Posture:** {st.session_state['poor_posture_count']} frames ({poor_percentage:.1f}%)")
            
            if poor_percentage > 50:
                st.error("⚠️ Action Required: High percentage of poor posture detected")
            elif poor_percentage > 30:
                st.warning("⚡ Improvement Needed: Moderate poor posture detected")
            else:
                st.success("✅ Excellent: Good posture maintained")
        
        with col2:
            st.subheader("Download Report")
            
            if st.button("📥 Generate PDF Report", type="primary"):
                with st.spinner("Generating PDF..."):
                    pdf_buffer = generate_pdf_report()
                    
                    st.download_button(
                        label="Download PDF",
                        data=pdf_buffer,
                        file_name=f"posture_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf"
                    )
                    st.success("Report generated successfully!")
