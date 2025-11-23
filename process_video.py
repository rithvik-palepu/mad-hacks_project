import os
import cv2
import json
import mediapipe as mp
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
ALLOWED_EXTENSIONS = {'webm', 'mp4'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize MediaPipe Face Mesh for ROI detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Specific landmark indices for the eye contours (used for bounding box calculation)
LEFT_EYE_LANDMARKS = [ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246 ]
RIGHT_EYE_LANDMARKS = [ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398 ]

# --- Utility Functions ---

def allowed_file(filename):
    """Checks if the file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def calculate_eye_roi(landmarks, eye_indices, width, height, padding=10):
    """
    Calculates the bounding box (ROI) coordinates for a given eye.
    
    Args:
        landmarks: The normalized landmark list from MediaPipe.
        eye_indices: The specific indices defining the eye contour.
        width: Frame width in pixels.
        height: Frame height in pixels.
        padding: Extra pixels to add around the calculated bounding box.
        
    Returns:
        A dictionary with the bounding box coordinates (x, y, w, h).
    """
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')

    # Convert normalized coordinates to pixel coordinates
    for index in eye_indices:
        lm = landmarks[index]
        x = int(lm.x * width)
        y = int(lm.y * height)
        
        if x < min_x: min_x = x
        if x > max_x: max_x = x
        if y < min_y: min_y = y
        if y > max_y: max_y = y
    
    # Apply padding
    min_x = max(0, min_x - padding)
    min_y = max(0, min_y - padding)
    max_x = min(width, max_x + padding)
    max_y = min(height, max_y + padding)

    # Calculate width and height of the box
    w = max_x - min_x
    h = max_y - min_y
    
    return {'x': min_x, 'y': min_y, 'w': w, 'h': h}


def process_video_for_roi(video_path):
    """
    Reads a video file, breaks it into frames, and finds the ROI for the eyes 
    in each frame using MediaPipe.
    
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

    # Initialize MediaPipe Face Mesh context manager
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break # End of video file

            # Convert the BGR image to RGB before processing
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process the image with Face Mesh
            results = face_mesh.process(image_rgb)
            
            current_frame_metrics = {
                'frame': frame_count,
                'timestamp_ms': (frame_count / frame_rate) * 1000,
                'left_roi': None,
                'right_roi': None
            }
            
            # Extract ROI if face is detected
            if results.multi_face_landmarks:
                # We only track the first face detected
                face_landmarks = results.multi_face_landmarks[0].landmark
                
                # Calculate and store ROI for both eyes
                left_roi = calculate_eye_roi(face_landmarks, LEFT_EYE_LANDMARKS, frame_width, frame_height)
                right_roi = calculate_eye_roi(face_landmarks, RIGHT_EYE_LANDMARKS, frame_width, frame_height)
                
                current_frame_metrics['left_roi'] = left_roi
                current_frame_metrics['right_roi'] = right_roi
                
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
        # Secure the filename to prevent path traversal issues
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            file.save(filepath)
            print(f"File saved successfully to: {filepath}")
        except Exception as e:
            return jsonify({"error": f"Failed to save file: {str(e)}"}), 500

        # 3. Process the video for ROI data
        # 
        roi_data, error = process_video_for_roi(filepath)
        
        # 4. Handle processing errors
        if error:
            # Clean up the file after a processing error
            os.remove(filepath)
            return jsonify({"error": error}), 500

        # 5. Success response (including cleanup)
        
        # NOTE: In a production environment, you would often save this JSON result 
        # to a database (like Firestore) instead of just returning it.
        
        # Clean up the large video file after successful processing
        os.remove(filepath)
        print(f"File cleaned up: {filepath}")
        
        # Return the structured analysis data
        return jsonify({
            "status": "success",
            "message": "Video processed and ROI data extracted for all frames.",
            "data": roi_data
        }), 200

    return jsonify({"error": "File type not allowed. Must be webm or mp4."}), 400

if __name__ == '__main__':
    # Running this locally requires installing Flask, OpenCV, and MediaPipe.
    # Use `flask run` or `python app.py` to start the server.
    # It will run on http://127.0.0.1:5000/ by default.
    app.run(debug=True)