# ████████      ██          ██          ████████ 
# ██            ██          ██          ██       
# ██  ████      ██          ██          ██  ████
# ██            ██          ██          ██       
# ████████      ████████    ████████    ████████ 

import cv2
import numpy as np

def preprocess_image(image):
    """
    Preprocess the image to match the model's input requirements.
    
    Steps:
    1. Convert to grayscale.
    2. Invert colors: background white, digit black.
    3. Resize to 28x28 pixels.
    4. Normalize pixel values.
    5. Reshape to (1, 28, 28, 1) for model input.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Invert colors: background white, digit black
    inverted = cv2.bitwise_not(gray)
    
    # Apply Gaussian Blur to smooth the image
    blurred = cv2.GaussianBlur(inverted, (5, 5), 0)
    
    # Threshold the image to make it binary
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours to crop the image to the digit
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get the bounding box of the largest contour
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        cropped = thresh[y:y+h, x:x+w]
    else:
        # If no contours found, use the thresholded image as is
        cropped = thresh
    
    # Resize to 20x20 while maintaining aspect ratio
    height, width = cropped.shape
    if height > width:
        new_height = 20
        new_width = int((width / height) * 20)
    else:
        new_width = 20
        new_height = int((height / width) * 20)
    
    resized = cv2.resize(cropped, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Create a 28x28 pixel image and paste the resized digit into the center
    padded = np.pad(resized, ((4,4),(4,4)), "constant", constant_values=0)
    
    # Ensure the image is 28x28
    final_image = cv2.resize(padded, (28,28), interpolation=cv2.INTER_AREA)
    
    # Normalize pixel values
    normalized = final_image.astype('float32') / 255.0
    
    # Reshape to match model input
    reshaped = normalized.reshape(1,28,28,1)
    
    return reshaped
