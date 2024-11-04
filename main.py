import cv2
import torch
import tkinter as tk
from tkinter import filedialog
import time
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load YOLOv5 Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Initialize GUI
def select_video_source():
    def start_detection(source):
        root.destroy()
        detect_and_track_objects(source)
    
    root = tk.Tk()
    root.title("Select Video Source")
    tk.Label(root, text="Choose source for object detection:").pack()
    tk.Button(root, text="Video File", command=lambda: start_detection(filedialog.askopenfilename())).pack()
    tk.Button(root, text="Webcam", command=lambda: start_detection(0)).pack()
    root.mainloop()

# Object Detection and Tracking Function
def detect_and_track_objects(source):
    cap = cv2.VideoCapture(source)
    prev_frame_time = 0
    fps_list, track_success_list = [], []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO detection
        results = model(frame, size=640)
        boxes = results.pandas().xyxy[0]  # Extract bounding boxes
        
        # Iterate over detected objects
        for index, row in boxes.iterrows():
            label = row['name']
            confidence = row['confidence']
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Calculate FPS
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time else 0
        prev_frame_time = new_frame_time
        fps_list.append(fps)
        
        # Display FPS on frame
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Show live feed
        cv2.imshow("Real-Time Object Detection and Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    # Performance Analysis
    avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0
    print(f"Average FPS: {avg_fps:.2f}")

    # Plot FPS over time
    plt.figure()
    plt.plot(fps_list, label='FPS over Time')
    plt.xlabel("Frame")
    plt.ylabel("FPS")
    plt.title("Frame Rate Performance")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    select_video_source()