# ████████      ██          ██          ████████ 
# ██            ██          ██          ██       
# ██  ████      ██          ██          ██  ████
# ██            ██          ██          ██       
# ████████      ████████    ████████    ████████ 

#!/usr/bin/env python

import sys
import os
from pathlib import Path
import tkinter as tk
from tkinter import *
from tkinter import ttk
from PIL import Image, ImageDraw, ImageTk
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import tempfile
import threading
import time  # Imported for tracking time

# Add the project's root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.preprocess import preprocess_image

# Define resource_path function to handle paths correctly for PyInstaller
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS  # PyInstaller extracts files to a temp folder _MEIPASS
    except Exception:
        # Change this to navigate out of 'app/' to the root folder
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    return os.path.join(base_path, relative_path)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Define the absolute path to model.h5 using resource_path
model_path = resource_path(os.path.join('model', 'model.h5'))

# Define the resampling filter
resample_filter = Image.Resampling.LANCZOS

# Debugging: Print the resolved model path to confirm it's correct.
print(f"Loading model from: {model_path}")

# Check if model_path exists
if not os.path.isfile(model_path):
    print(f"Error: Model file not found at {model_path}")
    sys.exit(1)
else:
    print(f"Model file found at {model_path}")

# Load the trained model
model = load_model(model_path)

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Digit Recognizer")
        self.geometry("500x600")  # Increased window size for better layout
        self.resizable(False, False)
        
        # Define styles
        self.style = ttk.Style(self)
        self.style.theme_use('clam')  # Choose a theme like 'clam', 'default', 'classic', etc.
        self.style.configure('TButton', font=('Helvetica', 12), padding=10)
        self.style.configure('TLabel', font=('Helvetica', 16))
        
        # Set background color
        self.configure(bg='#f0f0f0')  # Light grey background
        
        # Initialize drawing state
        self.drawing = False
        self.prediction_running = False
        self.last_draw_time = 0  # Timestamp of the last drawing action
        
        # Create main frames
        self.create_frames()
        
        # Create widgets
        self.create_canvas()
        self.create_control_panel()
        self.create_prediction_display()
        
        # Initialize prediction label to '0'
        self.label.config(text="0", foreground='black')
        
        # Start the periodic prediction check
        self.after(500, self.check_for_prediction)
    
    def create_frames(self):
        """Create and pack the main frames for the application."""
        # Frame for canvas
        self.canvas_frame = ttk.Frame(self, padding=10)
        self.canvas_frame.pack(fill='both', expand=True)
        
        # Frame for control buttons
        self.control_frame = ttk.Frame(self, padding=10)
        self.control_frame.pack(fill='x')
        
        # Frame for prediction display
        self.prediction_frame = ttk.Frame(self, padding=10)
        self.prediction_frame.pack(fill='x')
    
    def create_canvas(self):
        """Create and pack the drawing canvas."""
        # Create a canvas for drawing
        self.canvas = Canvas(self.canvas_frame, width=400, height=400, bg='white', cursor="cross")
        self.canvas.pack()
        
        # Initialize drawing
        self.image = Image.new("RGB", (400, 400), 'white')
        self.draw = ImageDraw.Draw(self.image)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
    
    def create_control_panel(self):
        """Create and center the control buttons."""
        # Configure grid columns for centering
        self.control_frame.columnconfigure(0, weight=1)
        self.control_frame.columnconfigure(1, weight=1)
        self.control_frame.columnconfigure(2, weight=1)
        
        # Clear Button with Icon
        clear_icon_path = resource_path(os.path.join('assets', 'clear.png'))  # Ensure you have an icon at this path
        if os.path.isfile(clear_icon_path):
            clear_icon = Image.open(clear_icon_path)
            clear_icon = clear_icon.resize((20, 20), resample=resample_filter)
            self.clear_photo = ImageTk.PhotoImage(clear_icon)
            clear_btn = ttk.Button(
                self.control_frame,
                text=' Clear',
                image=self.clear_photo,
                compound='left',
                command=self.clear
            )
        else:
            clear_btn = ttk.Button(self.control_frame, text='Clear', command=self.clear)
        
        # Place the Clear button in the center column
        clear_btn.grid(row=0, column=1, padx=10)
    
    def create_prediction_display(self):
        """Create and pack the prediction label."""
        self.label = ttk.Label(self.prediction_frame, text="0", foreground='black', background='#f0f0f0')
        self.label.pack()
    
    def paint(self, event):
        """Handle the painting on the canvas."""
        x, y = event.x, event.y
        r = 10  # Brush radius for better visibility
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill='black', outline='black')
        self.draw.ellipse([x-r, y-r, x+r, y+r], fill='black')
        
        # Update drawing flags and timestamp
        self.drawing = True
        self.last_draw_time = time.time()
    
    def on_release(self, event):
        """Handle the event when the mouse button is released."""
        self.drawing = False
    
    def clear(self):
        """Clear the canvas and reset the prediction label."""
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, 400, 400], fill='white')
        self.label.config(text="0", foreground='black')
    
    def check_for_prediction(self):
        """Periodically checks if the user is drawing or has drawn recently and triggers prediction accordingly."""
        current_time = time.time()
        time_since_last_draw = current_time - self.last_draw_time
        
        if (self.drawing or time_since_last_draw <= 1.0) and not self.prediction_running:
            self.prediction_running = True
            self.predict()
        
        self.after(500, self.check_for_prediction)
    
    def predict(self):
        """Initiates the prediction in a separate thread."""
        threading.Thread(target=self._predict, daemon=True).start()
    
    def _predict(self):
        """Handles the prediction logic."""
        try:
            # Save the canvas image to a temporary file
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                self.image.save(tmp.name)
                tmp_path = tmp.name

            # Read the image
            img = cv2.imread(tmp_path)
            if img is None:
                self.after(0, lambda: self.label.config(text="Error: Unable to read image.", foreground='red'))
                return

            # Preprocess the image
            processed = preprocess_image(img)

            # Predict
            prediction = model.predict(processed)
            digit = np.argmax(prediction)
            confidence = np.max(prediction)

            # Update label with prediction
            self.after(0, lambda: self.update_label(digit, confidence))

        except Exception as e:
            # Print detailed error messages to debug
            print(f"Prediction error: {e}")
            self.after(0, lambda: self.label.config(text="Prediction Error", foreground='red'))

        finally:
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)
            self.prediction_running = False
    
    def update_label(self, digit, confidence):
        """Updates the prediction label in the GUI."""
        self.label.config(text=f"Prediction: {digit} ({confidence*100:.2f}%)")
        if confidence < 0.5:
            self.label.config(foreground='red')
        elif confidence < 0.8:
            self.label.config(foreground='orange')
        else:
            self.label.config(foreground='green')

if __name__ == "__main__":
    app = App()
    app.mainloop()
