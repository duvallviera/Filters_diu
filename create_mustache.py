import cv2
import numpy as np

# Create a transparent image (RGBA)
width = 200
height = 50
mustache = np.zeros((height, width, 4), dtype=np.uint8)

# Draw mustache shape
cv2.ellipse(mustache, (width//2, height//2), (width//3, height//2), 0, 0, 360, (0, 0, 0, 255), -1)

# Save the mustache image
cv2.imwrite('mustache.png', mustache) 