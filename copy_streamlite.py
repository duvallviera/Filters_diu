import streamlit as st
import cv2
import numpy as np
import time
from PIL import Image
import io
import os
from datetime import datetime
import bluetooth
from bluetooth import *
import socket
import threading
import base64
import struct

# Bluetooth connection handler
class BluetoothCamera:
    def __init__(self):
        self.server_sock = None
        self.client_sock = None
        self.connected = False
        self.receiving = False
        self.frame_data = None
        
    def start_server(self):
        try:
            self.server_sock = BluetoothSocket(RFCOMM)
            self.server_sock.bind(("", PORT_ANY))
            self.server_sock.listen(1)
            
            port = self.server_sock.getsockname()[1]
            uuid = "94f39d29-7d6d-437d-973b-fba39e49d4ee"
            
            advertise_service(
                self.server_sock, "ImageTransferService",
                service_id=uuid,
                service_classes=[uuid, SERIAL_PORT_CLASS],
                profiles=[SERIAL_PORT_PROFILE]
            )
            
            st.info(f"Waiting for Bluetooth connection on RFCOMM channel {port}")
            return True
        except Exception as e:
            st.error(f"Failed to start Bluetooth server: {str(e)}")
            return False
    
    def accept_connection(self):
        try:
            self.client_sock, client_info = self.server_sock.accept()
            self.connected = True
            st.success(f"Accepted connection from {client_info}")
            return True
        except Exception as e:
            st.error(f"Failed to accept connection: {str(e)}")
            return False
    
    def start_receiving(self):
        self.receiving = True
        threading.Thread(target=self._receive_frames).start()
    
    def _receive_frames(self):
        while self.receiving and self.connected:
            try:
                # Receive frame size first
                size_data = self.client_sock.recv(8)
                frame_size = int.from_bytes(size_data, byteorder='big')
                
                # Receive frame data
                frame_data = b""
                while len(frame_data) < frame_size:
                    chunk = self.client_sock.recv(min(4096, frame_size - len(frame_data)))
                    if not chunk:
                        break
                    frame_data += chunk
                
                # Convert received data to image
                if len(frame_data) == frame_size:
                    img_array = np.frombuffer(base64.b64decode(frame_data), dtype=np.uint8)
                    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if frame is not None:
                        self.frame_data = frame
            except Exception as e:
                st.error(f"Error receiving frame: {str(e)}")
                break
    
    def get_frame(self):
        return self.frame_data
    
    def stop(self):
        self.receiving = False
        self.connected = False
        if self.client_sock:
            self.client_sock.close()
        if self.server_sock:
            self.server_sock.close()

# Initialize Bluetooth camera in session state
if 'bluetooth_camera' not in st.session_state:
    st.session_state.bluetooth_camera = BluetoothCamera()

# Network camera handler
class NetworkCamera:
    def __init__(self):
        self.server_sock = None
        self.client_sock = None
        self.connected = False
        self.receiving = False
        self.frame_data = None
        
    def start_server(self):
        try:
            self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_sock.bind(('0.0.0.0', 9090))
            self.server_sock.listen(1)
            
            st.info(f"Waiting for iPhone connection on port 9090...")
            return True
        except Exception as e:
            st.error(f"Failed to start server: {str(e)}")
            return False
    
    def accept_connection(self):
        try:
            self.client_sock, client_info = self.server_sock.accept()
            self.connected = True
            st.success(f"Accepted connection from {client_info}")
            return True
        except Exception as e:
            st.error(f"Failed to accept connection: {str(e)}")
            return False
    
    def start_receiving(self):
        self.receiving = True
        threading.Thread(target=self._receive_frames).start()
    
    def _receive_frames(self):
        while self.receiving and self.connected:
            try:
                # Receive frame size (4 bytes)
                size_data = self.client_sock.recv(4)
                if not size_data:
                    break
                frame_size = struct.unpack('>I', size_data)[0]
                
                # Receive frame data
                frame_data = b""
                while len(frame_data) < frame_size:
                    chunk = self.client_sock.recv(min(4096, frame_size - len(frame_data)))
                    if not chunk:
                        break
                    frame_data += chunk
                
                # Convert received data to image
                if len(frame_data) == frame_size:
                    img_array = np.frombuffer(frame_data, dtype=np.uint8)
                    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if frame is not None:
                        self.frame_data = frame
            except Exception as e:
                st.error(f"Error receiving frame: {str(e)}")
                break
    
    def get_frame(self):
        return self.frame_data
    
    def stop(self):
        self.receiving = False
        self.connected = False
        if self.client_sock:
            self.client_sock.close()
        if self.server_sock:
            self.server_sock.close()

# Initialize network camera in session state
if 'network_camera' not in st.session_state:
    st.session_state.network_camera = NetworkCamera()

# Set page config
st.set_page_config(
    page_title="Advanced Shape Analysis & Effects",
    page_icon="🎨",
    layout="wide"
)

def initialize_camera():
    """
    Try to initialize the camera with better error handling and fallback options.
    Returns the initialized camera or None if no camera is available.
    """
    try:
        # First try the default camera (index 0)
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Try DirectShow first
        if cap is not None and cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            return cap
        
        # If default camera fails, try other indices
        for camera_index in range(1, 3):  # Try indices 1 and 2
            if cap is not None:
                cap.release()  # Release previous capture object
            
            cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            if cap is not None and cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                return cap
        
        # If DirectShow failed, try without it
        for camera_index in range(3):
            if cap is not None:
                cap.release()
            
            cap = cv2.VideoCapture(camera_index)
            if cap is not None and cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                return cap
        
        return None
    except Exception as e:
        st.error(f"Camera initialization error: {str(e)}")
        return None
    finally:
        # Clean up if initialization failed
        if 'cap' in locals() and cap is not None and not cap.isOpened():
            cap.release()

# Title and description
st.title("Advanced Shape Analysis & Effects")
st.markdown("""
This application provides real-time shape analysis and various image effects.
Use the sliders below to adjust different parameters and see the results in real-time.
""")

# Create two columns for the interface
col1, col2 = st.columns([1, 2])

# Update slider values in session state
def update_slider(key):
    def callback(value):
        st.session_state[key] = value
    return callback

# Initialize session state variables
if 'camera' not in st.session_state:
    st.session_state.camera = None
if 'captured_image' not in st.session_state:
    st.session_state.captured_image = None
if 'processed_captured' not in st.session_state:
    st.session_state.processed_captured = None
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'video_writer' not in st.session_state:
    st.session_state.video_writer = None
if 'rotation_angle' not in st.session_state:
    st.session_state.rotation_angle = 0
if 'auto_enhance' not in st.session_state:
    st.session_state.auto_enhance = False
if 'blur' not in st.session_state:
    st.session_state.blur = 0
if 'pixelate' not in st.session_state:
    st.session_state.pixelate = 0
if 'brightness' not in st.session_state:
    st.session_state.brightness = 0
if 'contrast' not in st.session_state:
    st.session_state.contrast = 0
if 'edge' not in st.session_state:
    st.session_state.edge = 0
if 'mustache' not in st.session_state:
    st.session_state.mustache = False
if 'color_temperature' not in st.session_state:
    st.session_state.color_temperature = 0
if 'mustache_active' not in st.session_state:
    st.session_state.mustache_active = False
if 'smoothing_strength' not in st.session_state:
    st.session_state.smoothing_strength = 75
if 'blemish_threshold' not in st.session_state:
    st.session_state.blemish_threshold = 8
if 'sharpening_amount' not in st.session_state:
    st.session_state.sharpening_amount = 0.5
if 'vignette_strength' not in st.session_state:
    st.session_state.vignette_strength = 0.3
if 'gamma_value' not in st.session_state:
    st.session_state.gamma_value = 1.2
if 'saturation' not in st.session_state:
    st.session_state.saturation = 100
if 'sharpness' not in st.session_state:
    st.session_state.sharpness = 50

# Initialize visual effect variables
if 'glow' not in st.session_state:
    st.session_state.glow = 0
if 'color' not in st.session_state:
    st.session_state.color = 50
if 'artistic' not in st.session_state:
    st.session_state.artistic = 0
if 'motion' not in st.session_state:
    st.session_state.motion = 50
if 'texture' not in st.session_state:
    st.session_state.texture = 0
if 'emboss' not in st.session_state:
    st.session_state.emboss = 0
if 'cartoon' not in st.session_state:
    st.session_state.cartoon = 0
if 'thermal' not in st.session_state:
    st.session_state.thermal = 0
if 'night' not in st.session_state:
    st.session_state.night = 0
if 'vintage' not in st.session_state:
    st.session_state.vintage = 0
if 'hdr' not in st.session_state:
    st.session_state.hdr = 0
if 'transparency' not in st.session_state:
    st.session_state.transparency = 50
if 'contour' not in st.session_state:
    st.session_state.contour = 1000
if 'smooth' not in st.session_state:
    st.session_state.smooth = 20

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    
    # Auto-enhance toggle
    st.session_state.auto_enhance = st.checkbox("Auto Enhance", value=st.session_state.auto_enhance)
    
    # Reset button at the top
    if st.button("Reset All Settings"):
        # Basic settings
        st.session_state.blur = 0
        st.session_state.brightness = 0
        st.session_state.contrast = 0
        st.session_state.saturation = 100
        st.session_state.sharpness = 50
        st.session_state.edge = 50  # Reset edge to default 50
        st.session_state.color_temperature = 0
        st.session_state.rotation_angle = 0
        st.session_state.auto_enhance = False
        
        # Portrait enhancement settings
        st.session_state.gamma_value = 1.2
        st.session_state.smoothing_strength = 75
        st.session_state.blemish_threshold = 8
        st.session_state.sharpening_amount = 0.5
        st.session_state.vignette_strength = 0.3
        
        # Advanced settings
        st.session_state.contour = 1000
        st.session_state.smooth = 20
        st.session_state.glow = 0
        st.session_state.color = 50
        st.session_state.pixelate = 0
        st.session_state.artistic = 0
        st.session_state.motion = 50
        st.session_state.texture = 0
        st.session_state.emboss = 0
        st.session_state.cartoon = 0
        st.session_state.thermal = 0
        st.session_state.night = 0
        st.session_state.vintage = 0
        st.session_state.hdr = 0
        st.session_state.transparency = 50
        st.session_state.mustache_active = False
        
        st.rerun()
    
    # Create tabs for different settings
    tab1, tab2 = st.tabs(["Basic Settings", "Advanced Settings"])
    
    with tab1:
        st.subheader("Basic Image Adjustments")
        # Add color temperature slider
        temp_col1, temp_col2 = st.columns([3, 1])
        with temp_col1:
            st.session_state.color_temperature = st.slider("Color Temperature", -100, 100, 
                                                         st.session_state.get('color_temperature', 0),
                                                         help="Adjust color temperature (negative for cooler, positive for warmer)")
        with temp_col2:
            if st.button("↺", key="reset_temperature"):
                st.session_state.color_temperature = 0
                st.rerun()
        
        blur_col1, blur_col2 = st.columns([3, 1])
        with blur_col1:
            st.session_state.blur = st.slider("Background Blur", 0, 100, 
                                            st.session_state.get('blur', 0))
        with blur_col2:
            if st.button("↺", key="reset_blur"):
                st.session_state.blur = 0
                st.rerun()
        
        brightness_col1, brightness_col2 = st.columns([3, 1])
        with brightness_col1:
            st.session_state.brightness = st.slider("Brightness", -50, 50, 
                                                  st.session_state.get('brightness', 0))
        with brightness_col2:
            if st.button("↺", key="reset_brightness"):
                st.session_state.brightness = 0
                st.rerun()
        
        contrast_col1, contrast_col2 = st.columns([3, 1])
        with contrast_col1:
            st.session_state.contrast = st.slider("Contrast", -50, 50, 
                                                st.session_state.get('contrast', 0))
        with contrast_col2:
            if st.button("↺", key="reset_contrast"):
                st.session_state.contrast = 0
                st.rerun()
        
        saturation_col1, saturation_col2 = st.columns([3, 1])
        with saturation_col1:
            st.session_state.saturation = st.slider("Saturation", 0, 200, 
                                                  st.session_state.get('saturation', 100))
        with saturation_col2:
            if st.button("↺", key="reset_saturation"):
                st.session_state.saturation = 100
                st.rerun()
        
        sharpness_col1, sharpness_col2 = st.columns([3, 1])
        with sharpness_col1:
            st.session_state.sharpness = st.slider("Sharpness", 0, 100, 
                                                 st.session_state.get('sharpness', 50))
        with sharpness_col2:
            if st.button("↺", key="reset_sharpness"):
                st.session_state.sharpness = 50
                st.rerun()
        
        st.subheader("Portrait Enhancement")
        gamma_col1, gamma_col2 = st.columns([3, 1])
        with gamma_col1:
            st.session_state.gamma_value = st.slider("Gamma Correction", 0.5, 2.0, 
                                                   st.session_state.get('gamma_value', 1.2),
                                                   help="Adjust image brightness (>1 brightens, <1 darkens)")
        with gamma_col2:
            if st.button("↺", key="reset_gamma"):
                st.session_state.gamma_value = 1.2
                st.rerun()
        
        smooth_col1, smooth_col2 = st.columns([3, 1])
        with smooth_col1:
            st.session_state.smoothing_strength = st.slider("Skin Smoothing", 0, 150, 
                                                          st.session_state.get('smoothing_strength', 75),
                                                          help="Adjust skin smoothing intensity")
        with smooth_col2:
            if st.button("↺", key="reset_smooth"):
                st.session_state.smoothing_strength = 75
                st.rerun()
        
        blemish_col1, blemish_col2 = st.columns([3, 1])
        with blemish_col1:
            st.session_state.blemish_threshold = st.slider("Blemish Removal", 1, 20, 
                                                         st.session_state.get('blemish_threshold', 8),
                                                         help="Adjust blemish removal sensitivity")
        with blemish_col2:
            if st.button("↺", key="reset_blemish"):
                st.session_state.blemish_threshold = 8
                st.rerun()
        
        sharp_col1, sharp_col2 = st.columns([3, 1])
        with sharp_col1:
            st.session_state.sharpening_amount = st.slider("Sharpening", 0.0, 1.0, 
                                                         st.session_state.get('sharpening_amount', 0.5),
                                                         help="Adjust image sharpness")
        with sharp_col2:
            if st.button("↺", key="reset_sharp"):
                st.session_state.sharpening_amount = 0.5
                st.rerun()
        
        vignette_col1, vignette_col2 = st.columns([3, 1])
        with vignette_col1:
            st.session_state.vignette_strength = st.slider("Vignette", 0.0, 1.0, 
                                                         st.session_state.get('vignette_strength', 0.3),
                                                         help="Adjust vignette effect strength")
        with vignette_col2:
            if st.button("↺", key="reset_vignette"):
                st.session_state.vignette_strength = 0.3
                st.rerun()
        
        # Reset button for basic settings
        if st.button("Reset Basic Settings"):
            # Basic settings
            st.session_state.blur = 0
            st.session_state.brightness = 0
            st.session_state.contrast = 0
            st.session_state.saturation = 100
            st.session_state.sharpness = 50
            st.session_state.color_temperature = 0
            st.session_state.rotation_angle = 0
            st.session_state.auto_enhance = False
            
            # Portrait enhancement settings
            st.session_state.gamma_value = 1.1  # Reduced from 1.2 for more natural look
            st.session_state.smoothing_strength = 35  # Reduced from 75 for more subtle smoothing
            st.session_state.blemish_threshold = 5  # Reduced from 8 for more selective blemish removal
            st.session_state.sharpening_amount = 0.3  # Reduced from 0.5 for gentler sharpening
            st.session_state.vignette_strength = 0.15  # Reduced from 0.3 for lighter vignette
            
            st.rerun()
    
    with tab2:
        st.subheader("Shape Detection")
        edge_col1, edge_col2 = st.columns([3, 1])
        with edge_col1:
            st.session_state.edge = st.slider("Edge Detection", 0, 100, 
                            st.session_state.get('edge', 50))
        with edge_col2:
            if st.button("Reset Edge"):
                st.session_state.edge = 50
                st.rerun()
        
        contour_col1, contour_col2 = st.columns([3, 1])
        with contour_col1:
            st.session_state.contour = st.slider("Contour Size", 100, 5000, 
                              st.session_state.get('contour', 1000))
        with contour_col2:
            if st.button("Reset Contour"):
                st.session_state.contour = 1000
                st.rerun()
        
        smooth_col1, smooth_col2 = st.columns([3, 1])
        with smooth_col1:
            st.session_state.smooth = st.slider("Edge Smoothing", 1, 50, 
                             st.session_state.get('smooth', 20))
        with smooth_col2:
            if st.button("Reset Smooth"):
                st.session_state.smooth = 20
                st.rerun()
        
        st.subheader("Visual Effects")
        # Add mustache filter option
        mustache_col1, mustache_col2 = st.columns([3, 1])
        with mustache_col1:
            if st.button("Toggle Mustache"):
                st.session_state.mustache_active = not st.session_state.mustache_active
                if st.session_state.mustache_active:
                    # Load mustache image
                    mustache_path = "mustache.png"
                    if os.path.exists(mustache_path):
                        st.session_state.mustache = cv2.imread(mustache_path, cv2.IMREAD_UNCHANGED)
                        st.success("Mustache activated!")
                    else:
                        st.error("Mustache image not found!")
                        st.session_state.mustache_active = False
                else:
                    st.success("Mustache deactivated!")
        
        # Visual Effects sliders
        st.session_state.glow = st.slider("Glow Effect", 0, 100, 
                        st.session_state.get('glow', 0))
        st.session_state.color = st.slider("Color Enhancement", 0, 100, 
                         st.session_state.get('color', 50))
        st.session_state.pixelate = st.slider("Pixelation", 0, 100, 
                           st.session_state.get('pixelate', 0))
        st.session_state.artistic = st.slider("Artistic Effect", 0, 100, 
                           st.session_state.get('artistic', 0))
        st.session_state.motion = st.slider("Motion Detection", 0, 100, 
                         st.session_state.get('motion', 50))
        st.session_state.texture = st.slider("Texture Effect", 0, 100, 
                          st.session_state.get('texture', 0))
        st.session_state.emboss = st.slider("Emboss Effect", 0, 100, 
                         st.session_state.get('emboss', 0))
        st.session_state.cartoon = st.slider("Cartoon Effect", 0, 100, 
                          st.session_state.get('cartoon', 0))
        st.session_state.thermal = st.slider("Thermal Vision", 0, 100, 
                          st.session_state.get('thermal', 0))
        st.session_state.night = st.slider("Night Vision", 0, 100, 
                        st.session_state.get('night', 0))
        st.session_state.vintage = st.slider("Vintage Effect", 0, 100, 
                          st.session_state.get('vintage', 0))
        st.session_state.hdr = st.slider("HDR Effect", 0, 100, 
                       st.session_state.get('hdr', 0))
        st.session_state.transparency = st.slider("Mask Transparency", 0, 100, 
                               st.session_state.get('transparency', 50))
        
        # Reset button for advanced settings
        if st.button("Reset Advanced Settings"):
            # Advanced settings
            st.session_state.edge = 50
            st.session_state.contour = 1000
            st.session_state.smooth = 20
            st.session_state.glow = 0
            st.session_state.color = 50
            st.session_state.pixelate = 0
            st.session_state.artistic = 0
            st.session_state.motion = 50
            st.session_state.texture = 0
            st.session_state.emboss = 0
            st.session_state.cartoon = 0
            st.session_state.thermal = 0
            st.session_state.night = 0
            st.session_state.vintage = 0
            st.session_state.hdr = 0
            st.session_state.transparency = 50
            st.session_state.mustache_active = False
            
            st.rerun()
    
    # Camera controls
    st.subheader("Camera Controls")
    camera_col1, camera_col2 = st.columns(2)
    with camera_col1:
        if st.button("Start Camera"):
            if st.session_state.camera is None:
                with st.spinner("Initializing camera..."):
                    camera = initialize_camera()
                    if camera is not None and camera.isOpened():
                        st.session_state.camera = camera
                        st.success("Camera started successfully!")
                    else:
                        st.error("No camera found or failed to initialize. Please check your camera connection and try again.")
                        if st.session_state.camera is not None:
                            st.session_state.camera.release()
    
    with camera_col2:
        if st.button("Stop Camera"):
            if st.session_state.camera is not None:
                st.session_state.camera.release()
                st.session_state.camera = None
                st.success("Camera stopped.")
    
    # Network Camera Controls
    st.subheader("iPhone Camera Connection")
    net_col1, net_col2 = st.columns(2)
    
    with net_col1:
        if st.button("Connect iPhone Camera"):
            # Stop regular camera if running
            if st.session_state.camera is not None:
                st.session_state.camera.release()
                st.session_state.camera = None
            
            # Start network server
            if st.session_state.network_camera.start_server():
                st.info("Waiting for iPhone connection...")
                if st.session_state.network_camera.accept_connection():
                    st.session_state.network_camera.start_receiving()
                    st.success("iPhone camera connected!")
    
    with net_col2:
        if st.button("Disconnect iPhone"):
            if hasattr(st.session_state, 'network_camera'):
                st.session_state.network_camera.stop()
                st.session_state.network_camera = NetworkCamera()
                st.success("iPhone camera disconnected.")
    
    # Add connection instructions
    st.markdown("""
    ### Connect Your iPhone
    1. Install the iOS camera app on your iPhone
    2. Make sure your iPhone and Mac are on the same network
    3. Enter your Mac's IP address in the iOS app
    4. Click "Connect iPhone Camera" above
    """)
    
    # Get Mac's IP address
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    st.info(f"Your Mac's IP address is: {local_ip}")
    
    # Take picture button
    if st.button("Take Picture"):
        if st.session_state.camera is not None:
            ret, frame = st.session_state.camera.read()
            if ret:
                st.session_state.captured_image = frame.copy()
                st.success("Picture captured!")
    
    # Save options
    st.subheader("Save Options")
    if st.session_state.captured_image is not None:
        save_col1, save_col2 = st.columns(2)
        with save_col1:
            if st.button("Save as JPEG"):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"captured_image_{timestamp}.jpg"
                cv2.imwrite(filename, st.session_state.captured_image)
                st.success(f"Saved as {filename}")
        
        with save_col2:
            if st.button("Save as PNG"):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"captured_image_{timestamp}.png"
                cv2.imwrite(filename, st.session_state.captured_image)
                st.success(f"Saved as {filename}")
    
    # File uploader
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Main content area
with col1:
    st.subheader("Original")
    original_placeholder = st.empty()
    # Rotation controls for original
    original_rotation_col1, original_rotation_col2 = st.columns(2)
    with original_rotation_col1:
        if st.button("Original Rotate Left"):
            st.session_state.rotation_angle = (st.session_state.rotation_angle - 90) % 360
    with original_rotation_col2:
        if st.button("Original Rotate Right"):
            st.session_state.rotation_angle = (st.session_state.rotation_angle + 90) % 360

with col2:
    st.subheader("Processed")
    processed_placeholder = st.empty()
    # Rotation controls for processed
    processed_rotation_col1, processed_rotation_col2 = st.columns(2)
    with processed_rotation_col1:
        if st.button("Processed Rotate Left"):
            st.session_state.rotation_angle = (st.session_state.rotation_angle - 90) % 360
    with processed_rotation_col2:
        if st.button("Processed Rotate Right"):
            st.session_state.rotation_angle = (st.session_state.rotation_angle + 90) % 360
    
    # Capture processed image and video controls
    processed_capture_col1, processed_capture_col2 = st.columns(2)
    with processed_capture_col1:
        if st.button("Capture Processed Image"):
            if st.session_state.processed_captured is None:
                st.session_state.processed_captured = result.copy()
                st.success("Processed image captured!")
    
    with processed_capture_col2:
        if st.button("Start/Stop Recording"):
            if not st.session_state.recording:
                # Start recording
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"processed_video_{timestamp}.mp4"
                height, width = result.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                st.session_state.video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (width, height))
                st.session_state.recording = True
                st.success("Recording started!")
            else:
                # Stop recording
                if st.session_state.video_writer is not None:
                    st.session_state.video_writer.release()
                st.session_state.recording = False
                st.success("Recording stopped!")

# Image processing functions
def apply_texture_effect(image, intensity):
    if intensity == 0:
        return image
    texture = np.random.randint(0, 255, image.shape[:2], dtype=np.uint8)
    texture = cv2.GaussianBlur(texture, (5, 5), 0)
    alpha = intensity / 100.0
    return cv2.addWeighted(image, 1 - alpha, cv2.cvtColor(texture, cv2.COLOR_GRAY2BGR), alpha, 0)

def apply_emboss_effect(image, intensity):
    if intensity == 0:
        return image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.array([[-2,-1,0],
                      [-1, 1,1],
                      [ 0, 1,2]])
    emboss = cv2.filter2D(gray, -1, kernel)
    emboss = cv2.cvtColor(emboss, cv2.COLOR_GRAY2BGR)
    alpha = intensity / 100.0
    return cv2.addWeighted(image, 1 - alpha, emboss, alpha, 0)

def apply_cartoon_effect(image, intensity):
    if intensity == 0:
        return image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    color = cv2.bilateralFilter(image, 9, 250, 250)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 9, 2)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    alpha = intensity / 100.0
    return cv2.addWeighted(image, 1 - alpha, cartoon, alpha, 0)

def apply_thermal_effect(image, intensity):
    if intensity == 0:
        return image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    alpha = intensity / 100.0
    return cv2.addWeighted(image, 1 - alpha, thermal, alpha, 0)

def apply_night_vision(image, intensity):
    if intensity == 0:
        return image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    night = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    night[:,:,1] = gray
    night[:,:,0] = 0
    night[:,:,2] = 0
    noise = np.random.normal(0, 25, night.shape).astype(np.uint8)
    night = cv2.add(night, noise)
    alpha = intensity / 100.0
    return cv2.addWeighted(image, 1 - alpha, night, alpha, 0)

def apply_vintage_effect(image, intensity):
    if intensity == 0:
        return image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sepia = np.array([[0.393, 0.769, 0.189],
                     [0.349, 0.686, 0.168],
                     [0.272, 0.534, 0.131]])
    vintage = cv2.transform(image, sepia)
    rows, cols = image.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols, cols/4)
    kernel_y = cv2.getGaussianKernel(rows, rows/4)
    kernel = kernel_y * kernel_x.T
    mask = kernel / kernel.max()
    vintage = vintage * mask[:,:,np.newaxis]
    alpha = intensity / 100.0
    return cv2.addWeighted(image, 1 - alpha, vintage, alpha, 0)

def apply_hdr_effect(image, intensity):
    if intensity == 0:
        return image
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    hdr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    hdr = cv2.convertScaleAbs(hdr, alpha=1.2, beta=0)
    alpha = intensity / 100.0
    return cv2.addWeighted(image, 1 - alpha, hdr, alpha, 0)

def rotate_image(image, angle):
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated

def detect_edges(image, threshold):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, threshold, threshold * 2)
    return edges

def apply_mustache(image, mustache_img):
    if mustache_img is None:
        return image
    
    # Convert image to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Load face cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        # Calculate mustache position (below nose)
        mustache_width = int(w * 0.8)
        mustache_height = int(h * 0.2)
        mustache_x = x + int(w * 0.1)
        mustache_y = y + int(h * 0.6)
        
        # Resize mustache to fit face
        mustache_resized = cv2.resize(mustache_img, (mustache_width, mustache_height))
        
        # Create mask for mustache
        if mustache_resized.shape[2] == 4:  # If image has alpha channel
            mask = mustache_resized[:, :, 3] / 255.0
            mask = np.expand_dims(mask, axis=-1)
            mustache_rgb = mustache_resized[:, :, :3]
            
            # Blend mustache with image
            roi = image[mustache_y:mustache_y+mustache_height, mustache_x:mustache_x+mustache_width]
            blended = (mustache_rgb * mask + roi * (1 - mask)).astype(np.uint8)
            image[mustache_y:mustache_y+mustache_height, mustache_x:mustache_x+mustache_width] = blended
    
    return image

def apply_gamma_correction(img, gamma=1.0):
    """
    Light/Exposure correction via Gamma adjustment.
    Values > 1.0 brighten the image; < 1.0 darken it.
    """
    look_up_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                              for i in range(256)]).astype("uint8")
    return cv2.LUT(img, look_up_table)

def bilateral_smoothing(img, d=9, sigma_color=35, sigma_space=35):
    """
    Improved skin smoothing using bilateral filter with more natural parameters.
    """
    # Apply multiple passes with decreasing intensity for more natural result
    result = img.copy()
    for i in range(2):
        current_sigma = sigma_color * (1 - i * 0.3)  # Reduce intensity in subsequent passes
        result = cv2.bilateralFilter(result, d, current_sigma, sigma_space)
    return result

def remove_small_blemishes(img, size_threshold=5):
    """
    More selective blemish removal with improved detection.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Use local contrast enhancement to better detect blemishes
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Detect local intensity variations
    blur = cv2.GaussianBlur(enhanced, (5, 5), 0)
    diff = cv2.absdiff(enhanced, blur)
    
    # Threshold to find potential blemishes
    _, mask = cv2.threshold(diff, size_threshold * 2, 255, cv2.THRESH_BINARY)
    
    # Remove very small noise
    kernel = np.ones((2,2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find and filter contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_clean = np.zeros_like(mask)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5 and area < (size_threshold * size_threshold):
            cv2.drawContours(mask_clean, [cnt], 0, 255, -1)
    
    # Use inpainting with smaller radius for more natural results
    inpainted = cv2.inpaint(img, mask_clean, 2, cv2.INPAINT_TELEA)
    
    # Blend with original for more natural transition
    return cv2.addWeighted(img, 0.3, inpainted, 0.7, 0)

def sharpen_image(img, amount=0.3):
    """
    More natural sharpening with edge awareness.
    """
    # Convert to LAB color space for better edge detection
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel = lab[:,:,0]
    
    # Detect edges
    edges = cv2.Laplacian(l_channel, cv2.CV_64F).astype(np.float32)
    edge_mask = np.abs(edges) > 30
    
    # Create sharpening kernel
    kernel = np.array([
        [-0.5, -1, -0.5],
        [-1,   6,   -1],
        [-0.5, -1, -0.5]
    ], dtype=np.float32) * amount
    
    # Apply sharpening
    sharpened = cv2.filter2D(img, -1, kernel)
    
    # Blend based on edge mask
    result = img.copy()
    result[edge_mask] = sharpened[edge_mask]
    
    return result

def add_vignette(img, strength=0.15):
    """
    Add a subtle vignette effect with improved brightness preservation.
    """
    rows, cols = img.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols, cols / 2)
    kernel_y = cv2.getGaussianKernel(rows, rows / 2)
    kernel = kernel_y * kernel_x.T
    mask = kernel / kernel.max()
    mask = np.power(mask, 0.5)  # Reduce darkness by applying square root
    mask_3ch = np.dstack([mask, mask, mask])
    vignette = img.copy().astype(np.float32)
    # Use additive blending to preserve brightness
    vignette = vignette * (1.0 - strength + strength * mask_3ch)
    # Add a slight brightness boost to compensate for darkening
    vignette = vignette * (1.0 + strength * 0.2)
    return np.clip(vignette, 0, 255).astype(np.uint8)

def detect_portrait(image):
    """
    Detect if the image contains a portrait/face.
    Returns True if a face is detected, False otherwise.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return len(faces) > 0

def auto_enhance_image(image):
    """
    Enhanced portrait processing with automatic detection.
    """
    # Check if image contains a portrait
    is_portrait = detect_portrait(image)
    
    # Convert to float32 for better precision
    image = image.astype(np.float32) / 255.0
    image = (image * 255).astype(np.uint8)

    if is_portrait:
        # Portrait-specific enhancements
        # 1. Light correction via gamma
        corrected = apply_gamma_correction(image, gamma=1.2)  # Slightly brighter for portraits

        # 2. Remove small blemishes
        blemish_removed = remove_small_blemishes(corrected, size_threshold=8)

        # 3. Skin smoothing with moderate strength
        smoothed = bilateral_smoothing(blemish_removed, d=15, sigma_color=75, sigma_space=75)

        # 4. Subtle sharpening
        sharpened = sharpen_image(smoothed, amount=0.5)

        # 5. Add subtle vignette
        final = add_vignette(sharpened, strength=0.3)

    else:
        # General image enhancements
        # 1. Auto contrast using CLAHE
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        # 2. Enhance colors
        a = cv2.add(a, 3)  # Subtle color boost
        b = cv2.add(b, 3)
        
        # Merge channels
        enhanced = cv2.merge([l, a, b])
        final = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # 3. Subtle sharpening for non-portraits
        final = sharpen_image(final, amount=0.3)

    return final

def create_professional_lut():
    # Create a professional color grading LUT
    identity = np.arange(256, dtype=np.uint8)
    
    # Enhance shadows and highlights separately
    shadows = np.power(identity / 255.0, 0.9) * 255  # Lift shadows
    highlights = np.power(identity / 255.0, 0.95) * 255  # Preserve highlights
    
    # Blend shadows and highlights
    lut = np.clip(shadows * 0.7 + highlights * 0.3, 0, 255).astype(np.uint8)
    
    return lut

def adjust_color_temperature(image, temperature):
    """
    Adjust the color temperature of an image.
    Negative values make it cooler (more blue), positive values make it warmer (more orange).
    """
    if temperature == 0:
        return image
    
    # Convert to float32
    image_float = image.astype(np.float32)
    
    # Normalize temperature to a smaller range
    temp_factor = temperature / 100.0
    
    # Adjust color channels
    if temp_factor > 0:  # Warmer
        image_float[:, :, 0] = np.clip(image_float[:, :, 0] * (1 - temp_factor * 0.5), 0, 255)  # Reduce blue
        image_float[:, :, 2] = np.clip(image_float[:, :, 2] * (1 + temp_factor * 0.5), 0, 255)  # Increase red
    else:  # Cooler
        temp_factor = abs(temp_factor)
        image_float[:, :, 0] = np.clip(image_float[:, :, 0] * (1 + temp_factor * 0.5), 0, 255)  # Increase blue
        image_float[:, :, 2] = np.clip(image_float[:, :, 2] * (1 - temp_factor * 0.5), 0, 255)  # Reduce red
    
    return image_float.astype(np.uint8)

def process_image(image):
    """
    Process image with all effects based on session state values
    """
    # Store original image
    original = image.copy()
    result = original.copy()
    
    # Only apply effects if auto-enhance is enabled or if any effect value is different from default
    if (st.session_state.auto_enhance or
        st.session_state.gamma_value != 1.2 or
        st.session_state.smoothing_strength != 75 or
        st.session_state.blemish_threshold != 8 or
        st.session_state.sharpening_amount != 0.5 or
        st.session_state.vignette_strength != 0.3 or
        st.session_state.color_temperature != 0 or
        st.session_state.blur != 0 or
        st.session_state.brightness != 0 or
        st.session_state.contrast != 0 or
        st.session_state.saturation != 100 or
        st.session_state.sharpness != 50 or
        st.session_state.edge != 0 or
        st.session_state.glow != 0 or
        st.session_state.pixelate != 0 or
        st.session_state.texture != 0 or
        st.session_state.emboss != 0 or
        st.session_state.cartoon != 0 or
        st.session_state.thermal != 0 or
        st.session_state.night != 0 or
        st.session_state.vintage != 0 or
        st.session_state.hdr != 0 or
        st.session_state.mustache_active):
        
        # Apply auto-enhancement if enabled
        if st.session_state.auto_enhance:
            result = auto_enhance_image(result)
        
        # Apply Portrait Enhancement effects
        # 1. Gamma Correction
        if st.session_state.gamma_value != 1.2:  # Only apply if not at default
            result = apply_gamma_correction(result, gamma=st.session_state.gamma_value)
        
        # 2. Skin Smoothing
        if st.session_state.smoothing_strength != 75:  # Only apply if not at default
            sigma_color = st.session_state.smoothing_strength
            sigma_space = st.session_state.smoothing_strength
            result = bilateral_smoothing(result, d=15, sigma_color=sigma_color, sigma_space=sigma_space)
        
        # 3. Blemish Removal
        if st.session_state.blemish_threshold != 8:  # Only apply if not at default
            result = remove_small_blemishes(result, size_threshold=st.session_state.blemish_threshold)
        
        # 4. Sharpening
        if st.session_state.sharpening_amount != 0.5:  # Only apply if not at default
            result = sharpen_image(result, amount=st.session_state.sharpening_amount)
        
        # 5. Vignette
        if st.session_state.vignette_strength != 0.3:  # Only apply if not at default
            result = add_vignette(result, strength=st.session_state.vignette_strength)
        
        # Apply color temperature
        if st.session_state.color_temperature != 0:
            result = adjust_color_temperature(result, st.session_state.color_temperature)
        
        # Apply background blur
        if st.session_state.blur > 0:
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.dilate(edges, kernel, iterations=1)
            mask = cv2.GaussianBlur(mask, (5,5), 0)
            mask = mask.astype(np.float32) / 255.0
            mask = np.expand_dims(mask, axis=-1)
            blurred = cv2.GaussianBlur(result, (21,21), st.session_state.blur)
            result = result * (1 - mask) + blurred * mask
            result = result.astype(np.uint8)
        
        # Apply basic adjustments
        if st.session_state.brightness != 0:
            result = cv2.convertScaleAbs(result, alpha=1.0, beta=st.session_state.brightness)
        
        if st.session_state.contrast != 0:
            result = cv2.convertScaleAbs(result, alpha=1.0 + st.session_state.contrast/100.0, beta=0)
        
        if st.session_state.saturation != 100:  # Apply only if not at default
            hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
            hsv[:,:,1] = cv2.convertScaleAbs(hsv[:,:,1], alpha=st.session_state.saturation/100.0)
            result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        if st.session_state.sharpness != 50:  # Apply only if not at default
            amount = (st.session_state.sharpness - 50) / 50.0  # Convert to -1.0 to 1.0 range
            result = sharpen_image(result, amount=1.0 + amount)
        
        # Apply visual effects
        if st.session_state.edge > 0:
            edges = detect_edges(result, st.session_state.edge)
            edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            result = cv2.addWeighted(result, 0.7, edges_colored, 0.3, 0)
        
        if st.session_state.glow > 0:
            blurred = cv2.GaussianBlur(result, (21, 21), 5)
            result = cv2.addWeighted(result, 1.0, blurred, st.session_state.glow/100.0, 0)
        
        if st.session_state.pixelate > 0:
            h, w = result.shape[:2]
            factor = st.session_state.pixelate + 1
            temp = cv2.resize(result, (w//factor, h//factor), interpolation=cv2.INTER_LINEAR)
            result = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
        
        if st.session_state.texture > 0:
            result = apply_texture_effect(result, st.session_state.texture)
        
        if st.session_state.emboss > 0:
            result = apply_emboss_effect(result, st.session_state.emboss)
        
        if st.session_state.cartoon > 0:
            result = apply_cartoon_effect(result, st.session_state.cartoon)
        
        if st.session_state.thermal > 0:
            result = apply_thermal_effect(result, st.session_state.thermal)
        
        if st.session_state.night > 0:
            result = apply_night_vision(result, st.session_state.night)
        
        if st.session_state.vintage > 0:
            result = apply_vintage_effect(result, st.session_state.vintage)
        
        if st.session_state.hdr > 0:
            result = apply_hdr_effect(result, st.session_state.hdr)
        
        # Apply mustache if active
        if st.session_state.mustache_active and st.session_state.mustache is not None:
            result = apply_mustache(result, st.session_state.mustache)
        
        # Ensure both images are uint8 and have the same size and channels
        original = original.astype(np.uint8)
        result = result.astype(np.uint8)
        
        # Resize if dimensions don't match
        if result.shape != original.shape:
            result = cv2.resize(result, (original.shape[1], original.shape[0]))
        
        # Convert to BGR if grayscale
        if len(result.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        if len(original.shape) == 2:
            original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        
        # Final blend with original for subtle effect
        if st.session_state.transparency != 50:  # Only blend if transparency is not at default
            alpha = 1.0 - st.session_state.get('transparency', 50) / 100.0
            result = cv2.addWeighted(result, alpha, original, 1.0 - alpha, 0)
    
    return result

# Main loop
while True:
    if uploaded_file is not None:
        # Process uploaded image
        image = Image.open(uploaded_file)
        image = np.array(image)
        if len(image.shape) == 2:  # Grayscale image
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:  # RGBA image
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        
        # Apply rotation to original image
        if st.session_state.rotation_angle != 0:
            image = rotate_image(image, st.session_state.rotation_angle)
        
        # Display original
        original_placeholder.image(image, channels="BGR")
        
        # Process and display result
        result = process_image(image)
        processed_placeholder.image(result, channels="BGR")
        
        # Record processed frame if recording is active
        if st.session_state.recording and st.session_state.video_writer is not None:
            st.session_state.video_writer.write(result)
        
    elif st.session_state.captured_image is not None:
        # Apply rotation to captured image
        if st.session_state.rotation_angle != 0:
            rotated_captured = rotate_image(st.session_state.captured_image, st.session_state.rotation_angle)
        else:
            rotated_captured = st.session_state.captured_image
        
        # Display captured image
        original_placeholder.image(rotated_captured, channels="BGR")
        
        # Process and display result
        result = process_image(rotated_captured)
        processed_placeholder.image(result, channels="BGR")
        
        # Record processed frame if recording is active
        if st.session_state.recording and st.session_state.video_writer is not None:
            st.session_state.video_writer.write(result)
        
    elif st.session_state.camera is not None:
        ret, frame = st.session_state.camera.read()
        if ret:
            # Apply rotation to frame
            if st.session_state.rotation_angle != 0:
                frame = rotate_image(frame, st.session_state.rotation_angle)
            
            # Display original
            original_placeholder.image(frame, channels="BGR")
            
            # Process and display result
            result = process_image(frame)
            processed_placeholder.image(result, channels="BGR")
            
            # Record processed frame if recording is active
            if st.session_state.recording and st.session_state.video_writer is not None:
                st.session_state.video_writer.write(result)
            
            # Store frame for motion detection
            st.session_state.prev_frame = frame.copy()
    
    elif hasattr(st.session_state, 'network_camera') and st.session_state.network_camera.connected:
        # Get frame from Network camera
        frame = st.session_state.network_camera.get_frame()
        if frame is not None:
            # Apply rotation to frame
            if st.session_state.rotation_angle != 0:
                frame = rotate_image(frame, st.session_state.rotation_angle)
            
            # Display original
            original_placeholder.image(frame, channels="BGR")
            
            # Process and display result
            result = process_image(frame)
            processed_placeholder.image(result, channels="BGR")
            
            # Record processed frame if recording is active
            if st.session_state.recording and st.session_state.video_writer is not None:
                st.session_state.video_writer.write(result)
            
            # Store frame for motion detection
            st.session_state.prev_frame = frame.copy()
    
    time.sleep(0.1)  # Small delay to prevent overwhelming the UI

# Cleanup
if st.session_state.camera is not None:
    st.session_state.camera.release()
if st.session_state.video_writer is not None:
    st.session_state.video_writer.release() 