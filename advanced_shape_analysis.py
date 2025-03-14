import cv2
import numpy as np
import time
import argparse
import sys
import tkinter as tk
from tkinter import filedialog, ttk

# Set up Tkinter root for file dialogs and sliders
root = tk.Tk()
root.title("Advanced Shape Analysis and Effects")
root.geometry("400x800")  # Made taller to accommodate all settings

# Create main frame for all controls
main_frame = ttk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

# Create reset button at the top
def reset_all_sliders():
    # Basic Settings
    blur_var.set(25)
    brightness_var.set(50)
    contrast_var.set(50)
    saturation_var.set(50)
    hue_var.set(50)
    sharpness_var.set(50)
    
    # Advanced Settings
    edge_var.set(50)
    contour_var.set(1000)
    smooth_var.set(20)
    glow_var.set(0)
    color_var.set(50)
    pixelate_var.set(0)
    artistic_var.set(0)
    motion_var.set(50)
    texture_var.set(0)
    emboss_var.set(0)
    cartoon_var.set(0)
    thermal_var.set(0)
    night_var.set(0)
    vintage_var.set(0)
    hdr_var.set(0)
    transparency_var.set(50)

reset_button = ttk.Button(main_frame, text="Reset All Settings", command=reset_all_sliders)
reset_button.pack(pady=5)

# Create notebook for tabbed interface
notebook = ttk.Notebook(main_frame)
notebook.pack(fill=tk.BOTH, expand=True, pady=5)

# Create frames for basic and advanced settings
basic_frame = ttk.Frame(notebook)
advanced_frame = ttk.Frame(notebook)

# Add frames to notebook
notebook.add(basic_frame, text='Basic Settings')
notebook.add(advanced_frame, text='Advanced Settings')

# Create scrollable frames for both basic and advanced settings
def create_scrollable_frame(parent):
    canvas = tk.Canvas(parent)
    scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    return scrollable_frame

basic_scrollable = create_scrollable_frame(basic_frame)
advanced_scrollable = create_scrollable_frame(advanced_frame)

# Initialize all variables
# Basic Settings
blur_var = tk.IntVar(value=25)
brightness_var = tk.IntVar(value=50)
contrast_var = tk.IntVar(value=50)
saturation_var = tk.IntVar(value=50)
hue_var = tk.IntVar(value=50)
sharpness_var = tk.IntVar(value=50)

# Advanced Settings
edge_var = tk.IntVar(value=50)
contour_var = tk.IntVar(value=1000)
smooth_var = tk.IntVar(value=20)
glow_var = tk.IntVar(value=0)
color_var = tk.IntVar(value=50)
pixelate_var = tk.IntVar(value=0)
artistic_var = tk.IntVar(value=0)
motion_var = tk.IntVar(value=50)
texture_var = tk.IntVar(value=0)
emboss_var = tk.IntVar(value=0)
cartoon_var = tk.IntVar(value=0)
thermal_var = tk.IntVar(value=0)
night_var = tk.IntVar(value=0)
vintage_var = tk.IntVar(value=0)
hdr_var = tk.IntVar(value=0)
transparency_var = tk.IntVar(value=50)

# Create Basic Settings sliders
ttk.Label(basic_scrollable, text="Basic Image Adjustments", font=('Helvetica', 10, 'bold')).pack(fill=tk.X, pady=5)

basic_sliders = [
    ("Background Blur", blur_var, 3, 99),
    ("Brightness", brightness_var, 0, 100),
    ("Contrast", contrast_var, 0, 100),
    ("Saturation", saturation_var, 0, 100),
    ("Hue", hue_var, 0, 100),
    ("Sharpness", sharpness_var, 0, 100)
]

for name, var, min_val, max_val in basic_sliders:
    frame = ttk.Frame(basic_scrollable)
    frame.pack(fill=tk.X, pady=2)
    ttk.Label(frame, text=f"{name} ({min_val}-{max_val}):").pack(side=tk.LEFT)
    ttk.Scale(frame, from_=min_val, to=max_val, orient=tk.HORIZONTAL, variable=var).pack(side=tk.RIGHT, fill=tk.X, expand=True)

# Create Advanced Settings sliders
ttk.Label(advanced_scrollable, text="Shape Detection", font=('Helvetica', 10, 'bold')).pack(fill=tk.X, pady=5)

shape_sliders = [
    ("Edge Detection", edge_var, 0, 100),
    ("Contour Size", contour_var, 100, 5000),
    ("Edge Smoothing", smooth_var, 1, 50)
]

for name, var, min_val, max_val in shape_sliders:
    frame = ttk.Frame(advanced_scrollable)
    frame.pack(fill=tk.X, pady=2)
    ttk.Label(frame, text=f"{name} ({min_val}-{max_val}):").pack(side=tk.LEFT)
    ttk.Scale(frame, from_=min_val, to=max_val, orient=tk.HORIZONTAL, variable=var).pack(side=tk.RIGHT, fill=tk.X, expand=True)

ttk.Label(advanced_scrollable, text="Visual Effects", font=('Helvetica', 10, 'bold')).pack(fill=tk.X, pady=5)

effect_sliders = [
    ("Glow Effect", glow_var, 0, 100),
    ("Color Enhancement", color_var, 0, 100),
    ("Pixelation", pixelate_var, 0, 100),
    ("Artistic Effect", artistic_var, 0, 100),
    ("Motion Detection", motion_var, 0, 100),
    ("Texture Effect", texture_var, 0, 100),
    ("Emboss Effect", emboss_var, 0, 100),
    ("Cartoon Effect", cartoon_var, 0, 100),
    ("Thermal Vision", thermal_var, 0, 100),
    ("Night Vision", night_var, 0, 100),
    ("Vintage Effect", vintage_var, 0, 100),
    ("HDR Effect", hdr_var, 0, 100),
    ("Mask Transparency", transparency_var, 0, 100)
]

for name, var, min_val, max_val in effect_sliders:
    frame = ttk.Frame(advanced_scrollable)
    frame.pack(fill=tk.X, pady=2)
    ttk.Label(frame, text=f"{name} ({min_val}-{max_val}):").pack(side=tk.LEFT)
    ttk.Scale(frame, from_=min_val, to=max_val, orient=tk.HORIZONTAL, variable=var).pack(side=tk.RIGHT, fill=tk.X, expand=True)

def list_available_cameras(max_index=5):
    """Try opening camera indices from 0 to max_index-1 and return a list of indices that work."""
    available = []
    for index in range(max_index):
        cap = cv2.VideoCapture(index)
        if cap is not None:
            ret, _ = cap.read()
            if ret:
                available.append(index)
            cap.release()
    return available

def apply_texture_effect(image, intensity):
    """Apply texture effect to the image."""
    if intensity == 0:
        return image
    
    # Create texture pattern
    texture = np.random.randint(0, 255, image.shape[:2], dtype=np.uint8)
    texture = cv2.GaussianBlur(texture, (5, 5), 0)
    
    # Blend with original image
    alpha = intensity / 100.0
    return cv2.addWeighted(image, 1 - alpha, cv2.cvtColor(texture, cv2.COLOR_GRAY2BGR), alpha, 0)

def apply_emboss_effect(image, intensity):
    """Apply emboss effect to the image."""
    if intensity == 0:
        return image
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create emboss kernel
    kernel = np.array([[-2,-1,0],
                      [-1, 1,1],
                      [ 0, 1,2]])
    
    # Apply emboss effect
    emboss = cv2.filter2D(gray, -1, kernel)
    emboss = cv2.cvtColor(emboss, cv2.COLOR_GRAY2BGR)
    
    # Blend with original image
    alpha = intensity / 100.0
    return cv2.addWeighted(image, 1 - alpha, emboss, alpha, 0)

def apply_cartoon_effect(image, intensity):
    """Apply cartoon effect to the image."""
    if intensity == 0:
        return image
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter
    color = cv2.bilateralFilter(image, 9, 250, 250)
    
    # Apply adaptive thresholding
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 9, 2)
    
    # Combine color and edges
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    
    # Blend with original image
    alpha = intensity / 100.0
    return cv2.addWeighted(image, 1 - alpha, cartoon, alpha, 0)

def apply_thermal_effect(image, intensity):
    """Apply thermal effect to the image."""
    if intensity == 0:
        return image
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create thermal colormap
    thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    
    # Blend with original image
    alpha = intensity / 100.0
    return cv2.addWeighted(image, 1 - alpha, thermal, alpha, 0)

def apply_night_vision(image, intensity):
    """Apply night vision effect to the image."""
    if intensity == 0:
        return image
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply green tint
    night = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    night[:,:,1] = gray
    night[:,:,0] = 0
    night[:,:,2] = 0
    
    # Add noise
    noise = np.random.normal(0, 25, night.shape).astype(np.uint8)
    night = cv2.add(night, noise)
    
    # Blend with original image
    alpha = intensity / 100.0
    return cv2.addWeighted(image, 1 - alpha, night, alpha, 0)

def apply_vintage_effect(image, intensity):
    """Apply vintage effect to the image."""
    if intensity == 0:
        return image
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create sepia effect
    sepia = np.array([[0.393, 0.769, 0.189],
                     [0.349, 0.686, 0.168],
                     [0.272, 0.534, 0.131]])
    
    vintage = cv2.transform(image, sepia)
    
    # Add vignette effect
    rows, cols = image.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols, cols/4)
    kernel_y = cv2.getGaussianKernel(rows, rows/4)
    kernel = kernel_y * kernel_x.T
    mask = kernel / kernel.max()
    vintage = vintage * mask
    
    # Blend with original image
    alpha = intensity / 100.0
    return cv2.addWeighted(image, 1 - alpha, vintage, alpha, 0)

def apply_hdr_effect(image, intensity):
    """Apply HDR effect to the image."""
    if intensity == 0:
        return image
    
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    
    # Convert back to BGR
    hdr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Enhance contrast
    hdr = cv2.convertScaleAbs(hdr, alpha=1.2, beta=0)
    
    # Blend with original image
    alpha = intensity / 100.0
    return cv2.addWeighted(image, 1 - alpha, hdr, alpha, 0)

def detect_shapes(frame, edge_threshold, contour_threshold, smooth_size):
    """Detect and analyze shapes in the frame using advanced edge detection and contour finding."""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter for edge preservation
    smoothed = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Apply Canny edge detection with adaptive thresholding
    edges = cv2.Canny(smoothed, edge_threshold, edge_threshold * 2)
    
    # Apply morphological operations for better edge detection
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create mask and analyze shapes
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    shape_info = []
    
    # Create visualization image
    visualization = frame.copy()
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > contour_threshold:
            # Approximate the contour
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            cv2.drawContours(mask, [approx], -1, 255, -1)
            
            # Draw contour with different colors based on shape type
            analysis = analyze_shape_advanced(contour)
            shape_info.append(analysis)
            
            # Draw contour with color based on shape type
            color = (0, 255, 0)  # Default green
            if analysis['shape_type'] == "Circle":
                color = (255, 0, 0)  # Blue
            elif analysis['shape_type'] == "Rectangle":
                color = (0, 0, 255)  # Red
            elif analysis['shape_type'] == "Triangle":
                color = (255, 255, 0)  # Yellow
            elif analysis['shape_type'] == "Pentagon":
                color = (255, 0, 255)  # Magenta
            elif analysis['shape_type'] == "Hexagon":
                color = (0, 255, 255)  # Cyan
            
            # Draw contour with thickness based on area
            thickness = max(1, min(3, int(area / 1000)))
            cv2.drawContours(visualization, [approx], -1, color, thickness)
            
            # Draw shape type and metrics
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(visualization, analysis['shape_type'], (cx-20, cy),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Ensure smooth_size is odd
    if smooth_size % 2 == 0:
        smooth_size += 1
    
    # Apply smoothing to the mask
    mask = cv2.GaussianBlur(mask, (smooth_size, smooth_size), 0)
    
    return mask, shape_info, visualization

def analyze_shape_advanced(contour):
    """Perform advanced shape analysis."""
    # Basic metrics
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    # Advanced metrics
    if perimeter > 0:
        circularity = 4 * np.pi * area / (perimeter * perimeter)
    else:
        circularity = 0
    
    # Bounding rectangle
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h if h > 0 else 0
    
    # Convex hull
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 0
    
    # Minimum enclosing circle
    (x, y), radius = cv2.minEnclosingCircle(contour)
    circle_area = np.pi * radius * radius
    circularity_ratio = area / circle_area if circle_area > 0 else 0
    
    # Moments
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = 0, 0
    
    # Shape classification
    epsilon = 0.02 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    vertices = len(approx)
    
    # Determine shape type
    if vertices == 3:
        shape_type = "Triangle"
    elif vertices == 4:
        shape_type = "Rectangle"
    elif vertices == 5:
        shape_type = "Pentagon"
    elif vertices == 6:
        shape_type = "Hexagon"
    elif vertices > 6:
        shape_type = "Circle"
    else:
        shape_type = "Unknown"
    
    return {
        'area': area,
        'perimeter': perimeter,
        'circularity': circularity,
        'aspect_ratio': aspect_ratio,
        'solidity': solidity,
        'circularity_ratio': circularity_ratio,
        'centroid': (cx, cy),
        'vertices': vertices,
        'shape_type': shape_type,
        'bounding_box': (x, y, w, h)
    }

def detect_motion_advanced(frame, prev_frame, threshold):
    """Advanced motion detection with optical flow."""
    if prev_frame is None:
        return np.zeros(frame.shape[:2], dtype=np.uint8)
    
    # Convert frames to grayscale
    gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter for noise reduction
    gray1 = cv2.bilateralFilter(gray1, 9, 75, 75)
    gray2 = cv2.bilateralFilter(gray2, 9, 75, 75)
    
    # Calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    # Calculate magnitude and angle of flow
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # Create motion mask
    motion_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    motion_mask[magnitude > threshold] = 255
    
    # Apply morphological operations
    kernel = np.ones((5,5), np.uint8)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
    
    # Apply Gaussian blur for smoothness
    motion_mask = cv2.GaussianBlur(motion_mask, (5, 5), 0)
    
    return motion_mask

def enhance_sharpness(image, amount):
    """Enhance image sharpness with adaptive kernel."""
    kernel = np.array([[-1,-1,-1],
                      [-1, 9,-1],
                      [-1,-1,-1]]) * (amount / 100.0)
    return cv2.filter2D(image, -1, kernel)

def apply_glow_effect(image, intensity):
    """Apply a glow effect to the image."""
    if intensity == 0:
        return image
    
    # Create a blurred version of the image
    glow = cv2.GaussianBlur(image, (21, 21), 0)
    
    # Blend the original image with the glow
    alpha = intensity / 100.0
    return cv2.addWeighted(image, 1 + alpha, glow, alpha, 0)

def apply_pixelation(image, intensity):
    """Apply pixelation effect to the image."""
    if intensity == 0:
        return image
    
    # Calculate pixelation size based on intensity
    pixel_size = int(1 + (intensity / 10))
    
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Resize down
    temp = cv2.resize(image, (width // pixel_size, height // pixel_size), 
                      interpolation=cv2.INTER_LINEAR)
    
    # Resize back up
    pixelated = cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)
    
    return pixelated

def apply_color_enhancement(image, color, contrast, hue, saturation, brightness):
    """Apply color enhancements to the image."""
    # Convert to HSV for color manipulation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Adjust hue
    hsv[:,:,0] = (hsv[:,:,0] + (hue - 50) * 0.02) % 180
    
    # Adjust saturation
    hsv[:,:,1] = np.clip(hsv[:,:,1] * (1 + (saturation - 50) * 0.02), 0, 255)
    
    # Adjust brightness
    hsv[:,:,2] = np.clip(hsv[:,:,2] * (1 + (brightness - 50) * 0.02), 0, 255)
    
    # Convert back to BGR
    enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Apply contrast
    alpha = 1 + (contrast - 50) * 0.02
    beta = (brightness - 50) * 0.5
    enhanced = cv2.convertScaleAbs(enhanced, alpha=alpha, beta=beta)
    
    # Apply color enhancement
    enhanced = cv2.addWeighted(enhanced, 1 + (color - 50) * 0.02, image, 1, 0)
    
    return enhanced

def apply_artistic_effect(image, intensity):
    """Apply artistic effect to the image."""
    if intensity == 0:
        return image
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter for edge preservation
    bilateral = cv2.bilateralFilter(image, 9, 75, 75)
    
    # Create edge mask
    edges = cv2.Canny(gray, 50, 150)
    
    # Create artistic effect
    artistic = cv2.stylization(image, sigma_s=60, sigma_r=0.6)
    
    # Blend with original image
    alpha = intensity / 100.0
    result = cv2.addWeighted(image, 1 - alpha, artistic, alpha, 0)
    
    # Add edge emphasis
    result = cv2.addWeighted(result, 1, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), 0.1, 0)
    
    return result

def display_settings(frame, x_offset=0):
    """Display current settings on the frame."""
    y_pos = 60
    settings = [
        f"Blur: {blur_var.get()}",
        f"Edge: {edge_var.get()}",
        f"Glow: {glow_var.get()}",
        f"Contrast: {contrast_var.get()}",
        f"Brightness: {brightness_var.get()}",
        f"Transparency: {transparency_var.get()}"
    ]
    
    for setting in settings:
        cv2.putText(frame, setting, (10 + x_offset, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        y_pos += 20

# Set up command-line argument parsing
parser = argparse.ArgumentParser(description="Advanced Shape Analysis with Effects")
parser.add_argument("--camera", type=int, default=None,
                    help="Camera index to use (e.g., 0, 1, etc.)")
args, unknown = parser.parse_known_args()

# Camera selection code
camera_index = 0  # Explicitly use camera index 0
cap = cv2.VideoCapture(camera_index)

# Add a delay to allow camera to initialize
time.sleep(2)

if not cap.isOpened():
    print("Error: Could not open webcam with index", camera_index)
    sys.exit()

# Set camera properties for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

print("Webcam is running on camera index", camera_index)
print("Press 'q' to quit, 'j' to capture JPEG, 'p' to capture PNG, 'v' to toggle video recording.")

recording = False
video_writer = None
prev_frame = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break

    # Get current values from sliders
    blur_kernel = blur_var.get()
    edge_threshold = edge_var.get()
    sharpness = sharpness_var.get()
    contour_threshold = contour_var.get()
    smooth_size = smooth_var.get()
    glow_intensity = glow_var.get()
    color_intensity = color_var.get()
    contrast_intensity = contrast_var.get()
    hue_intensity = hue_var.get()
    saturation_intensity = saturation_var.get()
    brightness_intensity = brightness_var.get()
    pixelate_intensity = pixelate_var.get()
    artistic_intensity = artistic_var.get()
    texture_intensity = texture_var.get()
    emboss_intensity = emboss_var.get()
    cartoon_intensity = cartoon_var.get()
    thermal_intensity = thermal_var.get()
    night_intensity = night_var.get()
    vintage_intensity = vintage_var.get()
    hdr_intensity = hdr_var.get()
    transparency = transparency_var.get()

    # Ensure blur kernel is odd
    if blur_kernel % 2 == 0:
        blur_kernel += 1

    # Create blurred version of the frame
    blurred_frame = cv2.GaussianBlur(frame, (blur_kernel, blur_kernel), 0)
    
    # Detect and analyze shapes
    shape_mask, shape_info, visualization = detect_shapes(frame, edge_threshold, contour_threshold, smooth_size)
    
    # Detect motion with optical flow
    motion_mask = detect_motion_advanced(frame, prev_frame, motion_var.get())
    
    # Combine shape and motion masks
    combined_mask = cv2.bitwise_or(shape_mask, motion_mask)
    
    # Create a semi-transparent mask with color
    alpha = transparency / 100.0  # Transparency factor from slider
    mask_overlay = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)
    
    # Apply the mask to keep shapes sharp while background is blurred
    result = np.where(combined_mask[:, :, None] == 255, frame, blurred_frame)
    
    # Blend the mask with the result
    result = cv2.addWeighted(result, 1, mask_overlay, alpha, 0)
    
    # Blend with visualization
    result = cv2.addWeighted(result, 0.7, visualization, 0.3, 0)
    
    # Apply color enhancements
    result = apply_color_enhancement(result, color_intensity, contrast_intensity,
                                   hue_intensity, saturation_intensity, brightness_intensity)
    
    # Apply sharpness enhancement
    result = enhance_sharpness(result, sharpness)
    
    # Apply glow effect
    result = apply_glow_effect(result, glow_intensity)
    
    # Apply pixelation effect
    result = apply_pixelation(result, pixelate_intensity)
    
    # Apply artistic effect
    result = apply_artistic_effect(result, artistic_intensity)
    
    # Apply new effects
    result = apply_texture_effect(result, texture_intensity)
    result = apply_emboss_effect(result, emboss_intensity)
    result = apply_cartoon_effect(result, cartoon_intensity)
    result = apply_thermal_effect(result, thermal_intensity)
    result = apply_night_vision(result, night_intensity)
    result = apply_vintage_effect(result, vintage_intensity)
    result = apply_hdr_effect(result, hdr_intensity)

    # Display advanced shape analysis information
    if shape_info:
        for i, info in enumerate(shape_info):
            y_pos = 30 + i * 30
            cv2.putText(result, f"Shape {i+1}: {info['shape_type']}", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(result, f"Area: {info['area']:.0f} C: {info['circularity']:.2f}", 
                       (10, y_pos + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(result, f"Vertices: {info['vertices']} AR: {info['aspect_ratio']:.2f}", 
                       (10, y_pos + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display current settings
    display_settings(result)

    # If video recording is active, write the current frame
    if recording and video_writer is not None:
        video_writer.write(result)

    # Display both original and processed frames side by side
    # Resize frames to fit side by side
    height, width = frame.shape[:2]
    display_width = width // 2
    display_height = height
    
    # Resize both frames
    original_resized = cv2.resize(frame, (display_width, display_height))
    processed_resized = cv2.resize(result, (display_width, display_height))
    
    # Add labels to the frames
    cv2.putText(original_resized, "Original Video", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(processed_resized, "Processed Video", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display settings on both frames
    display_settings(original_resized)
    display_settings(processed_resized, display_width)
    
    # Create a combined window
    combined = np.hstack((original_resized, processed_resized))
    
    # Display the combined window
    cv2.imshow('Original | Processed', combined)
    
    # Update the Tkinter window
    root.update()
    
    # Store current frame for next iteration
    prev_frame = frame.copy()
    
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('j'):
        save_path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG files", "*.jpg"), ("All Files", "*.*")],
            title="Save JPEG Image"
        )
        if save_path:
            cv2.imwrite(save_path, result)
            print("Saved image as", save_path)
    elif key == ord('p'):
        save_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All Files", "*.*")],
            title="Save PNG Image"
        )
        if save_path:
            cv2.imwrite(save_path, result)
            print("Saved image as", save_path)
    elif key == ord('v'):
        if not recording:
            save_path = filedialog.asksaveasfilename(
                defaultextension=".mp4",
                filetypes=[("MP4 files", "*.mp4"), ("All Files", "*.*")],
                title="Save Video As"
            )
            if save_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                fps = 20.0
                frame_size = (result.shape[1], result.shape[0])
                video_writer = cv2.VideoWriter(save_path, fourcc, fps, frame_size)
                recording = True
                print("Started recording video to", save_path)
            else:
                print("Video recording cancelled.")
        else:
            recording = False
            video_writer.release()
            video_writer = None
            print("Stopped recording video")

# Clean up
cap.release()
if video_writer is not None:
    video_writer.release()
cv2.destroyAllWindows()
root.destroy()
print("Webcam closed.") 