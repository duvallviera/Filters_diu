import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import tempfile
import os
import mediapipe as mp

# Create application title and file uploader widgets
st.title("OpenCV Deep Learning based Face and Feature Detection")
st.sidebar.title("Settings")
input_type = st.sidebar.radio("Select Input Type", ["Image", "Video"])

# Add detection options in sidebar
st.sidebar.subheader("Detection Options")
detect_face = st.sidebar.checkbox("Detect Face", value=True)
detect_eyes = st.sidebar.checkbox("Detect Eyes")
detect_nose = st.sidebar.checkbox("Detect Nose")
detect_mouth = st.sidebar.checkbox("Detect Mouth")

if input_type == "Image":
    file_buffer = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
else:
    file_buffer = st.file_uploader("Upload a video", type=['mp4', 'avi', 'mov'])

# Initialize MediaPipe Face Mesh
@st.cache_resource
def load_face_mesh():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=0.5
    )
    return face_mesh

# Load Haar cascade classifiers
@st.cache_resource
def load_cascades():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    return face_cascade, eyes_cascade

# Function to get facial landmarks for nose and mouth
def get_face_landmarks(frame, face_mesh):
    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Process the image and get facial landmarks
    results = face_mesh.process(rgb_frame)
    
    features = {'nose': None, 'mouth': None}
    
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        
        # Get nose coordinates (nose tip is point 4)
        nose_tip = face_landmarks.landmark[4]
        h, w, _ = frame.shape
        nose_x = int(nose_tip.x * w)
        nose_y = int(nose_tip.y * h)
        nose_size = int(w * 0.1)  # Adjust size based on image width
        features['nose'] = (nose_x - nose_size//2, nose_y - nose_size//2, nose_size, nose_size)
        
        # Get mouth coordinates (upper lip is 13, lower lip is 14)
        upper_lip = face_landmarks.landmark[13]
        lower_lip = face_landmarks.landmark[14]
        mouth_x = int(upper_lip.x * w)
        mouth_y = int((upper_lip.y + lower_lip.y) * h / 2)
        mouth_w = int(w * 0.2)  # Adjust size based on image width
        mouth_h = int(abs(upper_lip.y - lower_lip.y) * h * 2)
        features['mouth'] = (mouth_x - mouth_w//2, mouth_y - mouth_h//2, mouth_w, mouth_h)
    
    return features

# Function for detecting faces and features in an image
def detectFeaturesHaar(frame, cascades, face_mesh, detect_options):
    face_cascade, eyes_cascade = cascades
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    features = {'faces': [], 'eyes': [], 'noses': [], 'mouths': []}
    
    try:
        # Detect faces
        if detect_options['face']:
            faces = face_cascade.detectMultiScale(frame_gray, 1.1, 4)
            features['faces'] = faces
            
            for (x, y, w, h) in faces:
                face_roi_gray = frame_gray[y:y+h, x:x+w]
                face_roi_color = frame[y:y+h, x:x+w]
                
                # Detect eyes within face region
                if detect_options['eyes']:
                    try:
                        # Adjust eye detection parameters
                        eyes = eyes_cascade.detectMultiScale(
                            face_roi_gray,
                            scaleFactor=1.1,
                            minNeighbors=5,
                            minSize=(int(w/10), int(h/10)),
                            maxSize=(int(w/4), int(h/4))
                        )
                        eyes = [(ex+x, ey+y, ew, eh) for (ex, ey, ew, eh) in eyes]
                        # Only keep the top two detected eyes
                        if len(eyes) > 2:
                            eyes = sorted(eyes, key=lambda e: e[1])[:2]
                        features['eyes'].extend(eyes)
                    except Exception as e:
                        st.warning(f"Warning: Error detecting eyes - {str(e)}")
        
        # Use MediaPipe for nose and mouth detection
        if detect_options['nose'] or detect_options['mouth']:
            face_features = get_face_landmarks(frame, face_mesh)
            if detect_options['nose'] and face_features['nose']:
                features['noses'].append(face_features['nose'])
            if detect_options['mouth'] and face_features['mouth']:
                features['mouths'].append(face_features['mouth'])
                
    except Exception as e:
        st.error(f"Error: Failed to detect features - {str(e)}")
        
    return features

# Function for drawing detected features
def draw_features(frame, features):
    # Draw faces (green)
    for (x, y, w, h) in features['faces']:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Draw eyes (blue)
    for (x, y, w, h) in features['eyes']:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # Draw noses (red)
    for (x, y, w, h) in features['noses']:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
    
    # Draw mouths (yellow)
    for (x, y, w, h) in features['mouths']:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
    
    return frame

# Function to process video
def process_video(video_path, cascades, face_mesh, detect_options):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    temp_dir = tempfile.mkdtemp()
    temp_output_path = os.path.join(temp_dir, "processed_video.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (frame_width, frame_height))
    
    progress_bar = st.progress(0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    current_frame = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        current_frame += 1
        progress_bar.progress(current_frame / frame_count)
        
        features = detectFeaturesHaar(frame, cascades, face_mesh, detect_options)
        processed_frame = draw_features(frame, features)
        
        out.write(processed_frame)
    
    cap.release()
    out.release()
    progress_bar.empty()
    
    return temp_output_path

# Function to generate a download link for output file.
def get_image_download_link(img, filename, text):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'
    return href

# Load cascades and face mesh
cascades = load_cascades()
face_mesh = load_face_mesh()

if file_buffer is not None:
    if input_type == "Image":
        # Read the file and convert it to opencv Image
        raw_bytes = np.asarray(bytearray(file_buffer.read()), dtype=np.uint8)
        image = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)

        # Create placeholders to display input and output images
        placeholders = st.columns(2)
        placeholders[0].image(image, channels='BGR')
        placeholders[0].text("Input Image")

        # Create detection options dictionary
        detect_options = {
            'face': detect_face,
            'eyes': detect_eyes,
            'nose': detect_nose,
            'mouth': detect_mouth
        }

        # Detect and draw features
        features = detectFeaturesHaar(image.copy(), cascades, face_mesh, detect_options)
        out_image = draw_features(image.copy(), features)

        # Display processed image
        placeholders[1].image(out_image, channels='BGR')
        placeholders[1].text("Output Image")

        # Convert opencv image to PIL
        out_image = Image.fromarray(out_image[:, :, ::-1])
        st.markdown(get_image_download_link(out_image, "face_output.jpg", 'Download Output Image'),
                    unsafe_allow_html=True)
    
    else:  # Video processing
        # Save uploaded video to temporary file
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, "input_video.mp4")
        with open(temp_path, "wb") as f:
            f.write(file_buffer.read())
            
        # Display original video
        st.video(temp_path)
        
        # Process video when button is clicked
        if st.button("Process Video"):
            detect_options = {
                'face': detect_face,
                'eyes': detect_eyes,
                'nose': detect_nose,
                'mouth': detect_mouth
            }
            
            st.text("Processing video... Please wait.")
            
            # Process the video
            output_path = process_video(temp_path, cascades, face_mesh, detect_options)
            
            # Display processed video
            st.text("Processing complete! Here's the result:")
            st.video(output_path)
            
            # Provide download link
            with open(output_path, 'rb') as f:
                video_bytes = f.read()
            st.download_button(
                label="Download processed video",
                data=video_bytes,
                file_name="processed_video.mp4",
                mime="video/mp4"
            )
            
            # Clean up temporary files
            os.remove(temp_path)
            os.remove(output_path)
            os.rmdir(temp_dir)
