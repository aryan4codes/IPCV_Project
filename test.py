import cv2
import torch
import streamlit as st
import time
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, deque
import numpy as np
import pandas as pd
from ultralytics import YOLO
import tempfile
import os
from deep_sort_realtime.deepsort_tracker import DeepSort
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class PerformanceMonitor:
    def __init__(self, window_size=50):
        self.window_size = window_size
        self.fps_data = deque(maxlen=window_size)
        self.objects_data = deque(maxlen=window_size)
        self.time_points = deque(maxlen=window_size)
        
        # Initialize the figure
        self.fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('FPS over Time', 'Objects Tracked over Time')
        )
        
        # Add traces
        self.fig.add_trace(
            go.Scatter(x=[], y=[], mode='lines', name='FPS',
                      line=dict(color='#2E86C1', width=2)),
            row=1, col=1
        )
        
        self.fig.add_trace(
            go.Scatter(x=[], y=[], mode='lines', name='Objects',
                      line=dict(color='#28B463', width=2)),
            row=1, col=2
        )
        
        # Update layout
        self.fig.update_layout(
            height=300,
            showlegend=False,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        self.fig.update_xaxes(title_text='Time', row=1, col=1)
        self.fig.update_xaxes(title_text='Time', row=1, col=2)
        self.fig.update_yaxes(title_text='FPS', row=1, col=1)
        self.fig.update_yaxes(title_text='Objects', row=1, col=2)

    def update(self, fps, num_objects):
        current_time = time.time()
        
        self.fps_data.append(fps)
        self.objects_data.append(num_objects)
        self.time_points.append(current_time)
        
        # Update the figure data
        self.fig.data[0].x = list(self.time_points)
        self.fig.data[0].y = list(self.fps_data)
        self.fig.data[1].x = list(self.time_points)
        self.fig.data[1].y = list(self.objects_data)

class ObjectAnalytics:
    def __init__(self):
        self.unique_objects = defaultdict(int)
        self.object_tracks = defaultdict(list)
        self.frame_detections = []
        self.object_positions = defaultdict(list)
        self.object_velocities = defaultdict(list)
        self.object_counts_per_frame = defaultdict(int)
        self.object_confidence_scores = defaultdict(list)
        self.tracking_history = defaultdict(list)
        self.counted_ids = set()  # Initialize a set to store counted track_ids
        
    def update(self, frame_num, tracks, class_names, confidence_scores):
        frame_objects = defaultdict(int)
        
        for track, class_name, conf in zip(tracks, class_names, confidence_scores):
            track_id = track.track_id
            bbox = track.to_tlbr()
            
            # Check if the track_id has already been counted
            if track_id not in self.counted_ids:
                # Update object counts
                self.unique_objects[class_name] += 1
                self.counted_ids.add(track_id)  # Mark this track_id as counted
            
            frame_objects[class_name] += 1
            
            # Store position and calculate velocity
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            self.object_positions[track_id].append((center_x, center_y))
            
            if len(self.object_positions[track_id]) >= 2:
                prev_pos = self.object_positions[track_id][-2]
                curr_pos = self.object_positions[track_id][-1]
                velocity = np.sqrt((curr_pos[0] - prev_pos[0])**2 + 
                                 (curr_pos[1] - prev_pos[1])**2)
                self.object_velocities[track_id].append(velocity)
            
            # Store confidence scores
            self.object_confidence_scores[class_name].append(conf)
            
            # Update tracking history
            self.tracking_history[track_id].append({
                'frame': frame_num,
                'bbox': bbox,
                'class': class_name,
                'confidence': conf
            })
        
        self.frame_detections.append({
            'frame_num': frame_num,
            'objects': dict(frame_objects)
        })

    def get_metrics(self):
        metrics = {
            'total_unique_objects': len(self.tracking_history),
            'objects_per_class': dict(self.unique_objects),
            'avg_confidence_per_class': {
                cls: np.mean(scores) 
                for cls, scores in self.object_confidence_scores.items()
            },
            'avg_velocity_per_track': {
                track_id: np.mean(velocities) 
                for track_id, velocities in self.object_velocities.items()
            },
            'tracking_persistence': {
                track_id: len(history) 
                for track_id, history in self.tracking_history.items()
            }
        }
        return metrics

    def reset_counts(self):
        self.unique_objects.clear()
        self.counted_ids.clear()
        self.object_tracks.clear()
        self.object_positions.clear()
        self.object_velocities.clear()
        self.object_confidence_scores.clear()
        self.tracking_history.clear()
        self.frame_detections.clear()

# Initialize models with caching
@st.cache_resource
def load_models():
    model = YOLO('yolov10n.pt')  # Ensure 'yolov5n.pt' is the correct model path
    tracker = DeepSort(
        max_age=30,
        n_init=3,
        max_cosine_distance=0.3,
        nn_budget=None,
        embedder="mobilenet",
        half=True,
        embedder_gpu=True
    )
    return model, tracker

def process_video(video_path, source_option):
    model, tracker = load_models()
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        st.error("Error opening video source.")
        return
    
    analytics = ObjectAnalytics()
    performance_metrics = defaultdict(list)
    frame_number = 0
    
    # Streamlit display elements
    frame_placeholder = st.empty()
    metrics_chart = st.empty()  # Placeholder for the charts
    progress_bar = st.progress(0)
    
    # Initialize performance monitor
    monitor = PerformanceMonitor()
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if source_option == "Video File" else None
    track_confidences = {}
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_number += 1
        start_time = time.time()
        
        # YOLO detection
        results = model(frame, verbose=False)
        
        # Process detections
        detections = []
        confidence_scores = []
        class_names = []
        
        if results and len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                
                w = x2 - x1
                h = y2 - y1
                
                # Store both detection and confidence
                detection = ([x1, y1, w, h], conf, class_name)
                detections.append(detection)
                confidence_scores.append(conf)
                class_names.append(class_name)
        
        # Update tracking
        if detections:
            tracks = tracker.update_tracks(detections, frame=frame)
            
            valid_tracks = []
            valid_classes = []
            valid_confidences = []
            
            for track, conf, class_name in zip(tracks, confidence_scores, class_names):
                if not track.is_confirmed():
                    continue
                    
                # Store the confidence score for this track
                track_confidences[track.track_id] = conf
                
                valid_tracks.append(track)
                valid_classes.append(class_name)
                valid_confidences.append(conf)
            
            if valid_tracks:
                analytics.update(frame_number, valid_tracks, valid_classes, valid_confidences)
            
            # Draw tracks and labels
            for track in valid_tracks:
                bbox = track.to_tlbr()
                
                # Get the stored confidence for this track
                conf = track_confidences.get(track.track_id, 0.0)
                class_name = next((c for t, c in zip(tracks, class_names) if t.track_id == track.track_id), "Unknown")
                
                # Draw bounding box
                cv2.rectangle(frame, 
                            (int(bbox[0]), int(bbox[1])), 
                            (int(bbox[2]), int(bbox[3])), 
                            (0, 255, 0), 2)
                
                # Draw label with track ID and confidence
                label = f"{class_name}-{track.track_id} ({conf:.2f})"
                cv2.putText(frame, label, 
                          (int(bbox[0]), int(bbox[1] - 10)),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Update performance metrics
        inference_time = time.time() - start_time
        current_fps = 1.0 / inference_time if inference_time > 0 else 0
        performance_metrics['inference_time'].append(inference_time)
        performance_metrics['fps'].append(current_fps)
        
        # Display frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
        
        # Update progress
        if total_frames:
            progress = frame_number / total_frames
            progress_bar.progress(min(progress, 1.0))
        
        # Update and display real-time metrics
        monitor.update(current_fps, len(analytics.tracking_history))
        metrics_chart.plotly_chart(monitor.fig, use_container_width=True)

    cap.release()
    return analytics, performance_metrics, frame_number

def create_average_confidence_visualization(analytics):
    metrics = analytics.get_metrics()
    avg_confidence = metrics['avg_confidence_per_class']
    
    conf_df = pd.DataFrame(list(avg_confidence.items()), columns=['Class', 'Average Confidence'])
    
    fig = px.bar(conf_df, x='Class', y='Average Confidence',
                 title="Average Confidence per Class",
                 labels={'Average Confidence': 'Average Confidence Score'},
                 color='Class',
                 color_discrete_sequence=px.colors.qualitative.Pastel)
    
    st.plotly_chart(fig, use_container_width=True)

def create_visualizations(analytics, performance_metrics, frame_number):
    st.header("📊 Detailed Analysis")
    
    
    metrics = analytics.get_metrics()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Unique Objects", metrics['total_unique_objects'])
    with col2:
        st.metric("Average FPS", f"{np.mean(performance_metrics['fps']):.2f}")
    with col3:
        st.metric("Average Inference Time (ms)", 
                 f"{np.mean(performance_metrics['inference_time'])*1000:.2f}")
    
    
    st.subheader("Object Class Distribution")
    class_dist = pd.DataFrame(list(metrics['objects_per_class'].items()),
                            columns=['Class', 'Count'])
    fig = px.bar(class_dist, x='Class', y='Count',
                 title="Objects Detected by Class",
                 color='Class',
                 color_discrete_sequence=px.colors.qualitative.Vivid)
    st.plotly_chart(fig)
    
    
    st.subheader("Confidence Score Distribution")
    conf_data = []
    for cls, scores in analytics.object_confidence_scores.items():
        conf_data.extend([(cls, score) for score in scores])
    conf_df = pd.DataFrame(conf_data, columns=['Class', 'Confidence'])
    fig = px.box(conf_df, x='Class', y='Confidence',
                 title="Confidence Score Distribution by Class",
                 color='Class',
                 color_discrete_sequence=px.colors.qualitative.Vivid)
    st.plotly_chart(fig)
    
    
    st.subheader("Average Confidence per Class")
    create_average_confidence_visualization(analytics)
    
    
    st.subheader("Object Tracking Analysis")
    track_data = []
    for track_id, history in analytics.tracking_history.items():
        track_data.append({
            'Track ID': track_id,
            'Frames Tracked': len(history),
            'Average Confidence': np.mean([h['confidence'] for h in history]),
            'Class': history[0]['class']
        })
    
    if track_data:  
        track_df = pd.DataFrame(track_data)
        fig = px.scatter(track_df, x='Frames Tracked', y='Average Confidence',
                        color='Class', title="Object Tracking Performance",
                        color_discrete_sequence=px.colors.qualitative.Vivid)
        st.plotly_chart(fig)
    else:
        st.warning("No tracking data available for visualization")
        track_df = pd.DataFrame()
    
    
    st.subheader("Performance Metrics Over Time")
    perf_df = pd.DataFrame({
        'Frame': range(1, frame_number + 1),
        'FPS': performance_metrics['fps'],
        'Inference Time (ms)': np.array(performance_metrics['inference_time']) * 1000
    })
    fig = px.line(perf_df, x='Frame', y=['FPS', 'Inference Time (ms)'],
                  title="Performance Metrics Over Time",
                  labels={'value': 'Metric Value', 'variable': 'Metric'})
    st.plotly_chart(fig)
    
    return track_df, perf_df

def main():
    st.title("🎥 Advanced Object Detection - Aryan Rajpurkar ( D019 - 60009220144 )")
    st.write("""
    Enhanced real-time object detection and tracking system with detailed analytics.
    IPCV Mini Project
    """)
    
    st.sidebar.header("🛠️ Configuration")
    source_option = st.sidebar.selectbox("Select Video Source", 
                                       ("Video File", "Webcam"))
    
    if source_option == "Video File":
        video_file = st.sidebar.file_uploader("Upload Video", 
                                            type=["mp4", "avi", "mov", "mkv"])
        if video_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())
            video_path = tfile.name
        else:
            st.sidebar.warning("Please upload a video file.")
            return
    else:
        video_path = 0  
    
    if st.sidebar.button("Start Analysis"):
        with st.spinner("Processing video..."):
            analytics, performance_metrics, frame_number = process_video(
                video_path, source_option
            )
            
            track_df, perf_df = create_visualizations(
                analytics, performance_metrics, frame_number
            )
            
            
            st.subheader("📤 Export Analysis")
            
            
            if not track_df.empty:
                csv_track = track_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Tracking Data (CSV)",
                    data=csv_track,
                    file_name='tracking_analysis.csv',
                    mime='text/csv',
                )
            
            
            csv_perf = perf_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Performance Data (CSV)",
                data=csv_perf,
                file_name='performance_metrics.csv',
                mime='text/csv',
            )
        
        
        if source_option == "Video File" and video_file is not None:
            os.unlink(video_path)

if __name__ == "__main__":
    main()
