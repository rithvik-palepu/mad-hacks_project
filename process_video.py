import os
import cv2
import json
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
ALLOWED_EXTENSIONS = {'webm', 'mp4'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load a reliable OpenCV Deep Neural Network (DNN) Face Detector
# NOTE: In a real app, you would load a model for eye landmarks. 
# For simplicity and stability, we use a robust face detector here to find the general ROI.
# We'll use a placeholder for a hypothetical Eye-Landmark Model since MediaPipe is failing.
print("NOTE: Using OpenCV placeholder for Face/Eye Detection due to MediaPipe dependency issues.")

# --- Utility Functions ---

def allowed_file(filename):
    """Checks if the file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def calculate_roi_from_face(face_bbox, width, height, scale_factor=0.3):
    """
    Calculates the general ROI (Region of Interest) for the eyes based on 
    the face bounding box.

    Args:
        face_bbox: [x, y, w, h] of the detected face.
        scale_factor: The estimated vertical position of the eyes relative to the face box (0.3 = 30% down).
    
    Returns:
        A dictionary with the bounding box coordinates for the general eye area.
    """
    x, y, w, h = face_bbox
    
    # Heuristic: Eyes are usually in the top 30-50% of the face box.
    # We'll isolate the top half of the face box as a general "Eye Area ROI".
    eye_area_x = x
    eye_area_y = y
    eye_area_w = w
    eye_area_h = int(h * 0.5) # Use the top half of the face box
    
    return {
        'x': eye_area_x, 
        'y': eye_area_y, 
        'w': eye_area_w, 
        'h': eye_area_h
    }


def process_video_for_roi(video_path):
    """
    Reads a video file, breaks it into frames, and finds the ROI for the eyes 
    in each frame using a simplified OpenCV approach.
    
    Args:
        video_path (str): Full path to the video file.
        
    Returns:
        dict: A dictionary containing frame-by-frame ROI data and video metadata.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "Error: Could not open video file."

    # Get video metadata
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video opened: {frame_width}x{frame_height} @ {frame_rate} FPS, {total_frames} frames.")

    frame_data = []
    frame_count = 0
    
    # Initialize a simple face detector (Haar cascade for maximum compatibility)
    # NOTE: In a professional app, you'd use a more accurate DNN model.
    # We use this for reliable installation and ROI simulation.
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break # End of video file

        # Convert the BGR image to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        current_frame_metrics = {
            'frame': frame_count,
            'timestamp_ms': (frame_count / frame_rate) * 1000,
            'face_detected': False,
            'eye_area_roi': None,
        }
        
        # Extract ROI if face is detected
        if len(faces) > 0:
            current_frame_metrics['face_detected'] = True
            # We only process the largest face (assuming the patient's face)
            x, y, w, h = faces[0] 
            
            # Calculate the general eye area ROI
            eye_area_roi = calculate_roi_from_face((x, y, w, h), frame_width, frame_height)
            
            current_frame_metrics['eye_area_roi'] = eye_area_roi
            
            # Optional: Draw the detection on the image for debugging (local save only)
            # cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # cv2.rectangle(image, (eye_area_roi['x'], eye_area_roi['y']), 
            #               (eye_area_roi['x'] + eye_area_roi['w'], eye_area_roi['y'] + eye_area_roi['h']), (0, 255, 255), 2)
            
        frame_data.append(current_frame_metrics)
        frame_count += 1

    cap.release()
    
    # Final structured output
    output_result = {
        'metadata': {
            'filepath': video_path,
            'frame_rate': frame_rate,
            'total_frames': total_frames,
            'dimensions': f"{frame_width}x{frame_height}",
            'processed_frames': frame_count
        },
        'roi_analysis': frame_data
    }
    
    return output_result, None

# --- Flask API Endpoint ---

@app.route('/api/process-video', methods=['POST'])
def process_video():
    """Receives a video file, processes it, and returns the ROI data."""
    
    # 1. Check for file presence
    if 'video_file' not in request.files:
        return jsonify({"error": "No video_file part in the request"}), 400
    
    file = request.files['video_file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    # 2. Save the file and check extension
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            file.save(filepath)
            print(f"File saved successfully to: {filepath}")
        except Exception as e:
            return jsonify({"error": f"Failed to save file: {str(e)}"}), 500

        # 3. Process the video for ROI data
        roi_data, error = process_video_for_roi(filepath)
        
        # 4. Handle processing errors
        if error:
            os.remove(filepath)
            return jsonify({"error": error}), 500

        # 5. Success response (including cleanup)
        os.remove(filepath)
        print(f"File cleaned up: {filepath}")
        
        # Return the structured analysis data
        return jsonify({
            "status": "success",
            "message": "Video processed and ROI data extracted for all frames using OpenCV.",
            "data": roi_data
        }), 200

    return jsonify({"error": "File type not allowed. Must be webm or mp4."}), 400

if __name__ == '__main__':
    # Running this locally requires installing Flask and OpenCV.
    app.run(debug=True)