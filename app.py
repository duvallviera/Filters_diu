import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

st.title("Interactive Image Processing App")

st.sidebar.header("Upload and Parameters")
uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is None:
    st.warning("Please upload an image file to proceed.")
    st.stop()

# Read image from uploaded file
file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
if image is None:
    st.error("Error reading the image!")
    st.stop()

# Convert BGR to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Sidebar sliders for interactive parameters
face_scale_x10 = st.sidebar.slider("Face Scale x10", 10, 20, 11, step=1)
min_neighbors = st.sidebar.slider("Min Neighbors", 3, 10, 5, step=1)
face_expansion = st.sidebar.slider("Face Expansion", 0, 50, 10, step=1)
blur_kernel = st.sidebar.slider("Blur Kernel (odd)", 1, 31, 5, step=2)
edge_thresh = st.sidebar.slider("Edge Sensibility", 50, 150, 50, step=1)

face_scale = face_scale_x10 / 10.0

# ---------------------------
# (A) Face Detection & Edge Overlay
# ---------------------------
# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=face_scale, minNeighbors=min_neighbors)
face_overlay = image_rgb.copy()

for (x, y, w, h) in faces:
    # Expand the bounding box
    x_exp = max(x - face_expansion, 0)
    y_exp = max(y - face_expansion, 0)
    x2 = min(x + w + face_expansion, image_rgb.shape[1])
    y2 = min(y + h + face_expansion, image_rgb.shape[0])
    cv2.rectangle(face_overlay, (x_exp, y_exp), (x2, y2), (0, 255, 0), 2)
    # Edge detection on face ROI
    face_roi = gray[y_exp:y2, x_exp:x2]
    edges = cv2.Canny(face_roi, edge_thresh, edge_thresh * 3)
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    # Blend edges with original face ROI (20% edge contribution)
    face_overlay[y_exp:y2, x_exp:x2] = cv2.addWeighted(face_overlay[y_exp:y2, x_exp:x2], 0.8, edges_color, 0.2, 0)

# ---------------------------
# (B) Composite (Blurred Background via GrabCut)
# ---------------------------
mask = np.zeros(image.shape[:2], np.uint8)
height, width = image.shape[:2]
# Define an initial rectangle (assumed to roughly cover the foreground)
rect = (int(width * 0.1), int(height * 0.1), int(width * 0.8), int(height * 0.8))
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)
cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
# Create mask: foreground pixels = 1, background = 0
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

# Apply Gaussian blur (ensure kernel size is odd)
blurred = cv2.GaussianBlur(image_rgb, (blur_kernel, blur_kernel), 0)
composite = image_rgb.copy()
composite[mask2 == 0] = blurred[mask2 == 0]

# ---------------------------
# (C) Sharpened Image (using fixed kernel)
# ---------------------------
sharpen_kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
sharpened = cv2.filter2D(image_rgb, ddepth=-1, kernel=sharpen_kernel)

# ---------------------------
# Display results using matplotlib subplots
# ---------------------------
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

axs[0, 0].imshow(image_rgb)
axs[0, 0].set_title("Original Image")
axs[0, 0].axis("off")

axs[0, 1].imshow(composite)
axs[0, 1].set_title("Composite (Blurred BG)")
axs[0, 1].axis("off")

axs[1, 0].imshow(face_overlay)
axs[1, 0].set_title("Face Detection & Edges")
axs[1, 0].axis("off")

axs[1, 1].imshow(sharpened)
axs[1, 1].set_title("Sharpened Image")
axs[1, 1].axis("off")

st.pyplot(fig)
