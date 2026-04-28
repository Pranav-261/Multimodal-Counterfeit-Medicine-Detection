# -*- coding: utf-8 -*-
"""
app_targeted_ocr.py

Flask web application using Targeted OCR (detect regions, then OCR crops).
Runs TensorFlow model on CPU.
"""

# --- Force CPU Usage for TensorFlow ---
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# -------------------------------------

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
import cv2
import numpy as np
import pandas as pd
import easyocr # Using EasyOCR for both detection and recognition
import time
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import logging
import math # For coordinate math

# =============================================================================
#                           --- Configuration ---
# =============================================================================
logging.basicConfig(level=logging.INFO)

# --- Model & Data Constants ---
# (Keep these the same)
IMG_WIDTH = 224; IMG_HEIGHT = 224; IMG_CHANNELS = 3; MAX_TEXT_LENGTH = 50
CLASS_NAMES = ['Counterfeit', 'Authentic']
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# --- Paths ---
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.join(APP_ROOT, "Caro_Laptop_Files")
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'uploads')
MODEL_FILENAME = r"mm_cmds_model_weighted_final.keras" # Use the augmented .h5 model
MODEL_PATH = os.path.join(BASE_PATH, MODEL_FILENAME)
TRAIN_DIR = os.path.join(BASE_PATH, "train")
TRAIN_CSV_OCR = os.path.join(TRAIN_DIR, "_classes_with_ocr.csv")

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Targeted OCR Config ---
MIN_BOX_AREA = 50 # Ignore detected boxes smaller than this area (pixels) - adjust as needed
CROP_PADDING = 3   # Add a small pixel padding around crops before OCR

# =============================================================================
#            --- Initialize App, Load Model & Char Map ---
# =============================================================================
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# --- Load Character Map ---
# (Same as before)
char_map = {}; text_vocab_size = 1
logging.info("--- Reconstructing Character Map ---")
# ... (char_map reconstruction code remains the same) ...
if os.path.exists(TRAIN_CSV_OCR):
    try:
        df_train = pd.read_csv(TRAIN_CSV_OCR); df_train['extracted_text'] = df_train['extracted_text'].fillna('')
        train_texts = df_train['extracted_text'].astype(str).tolist(); all_train_text = "".join(train_texts)
        unique_chars = sorted(list(set(all_train_text))); char_map = {char: i+1 for i, char in enumerate(unique_chars)}
        text_vocab_size = len(char_map) + 1; logging.info(f"Character map reconstructed. Vocab size: {text_vocab_size}")
    except Exception as e: logging.error(f"Error reconstructing char_map: {e}", exc_info=True)
else: logging.error(f"Training CSV ({TRAIN_CSV_OCR}) not found.")

# --- Load Keras Model ---
# (Same as before)
model = None
logging.info(f"--- Loading Keras Model from {MODEL_PATH} ---")
# ... (model loading code remains the same) ...
if not os.path.exists(MODEL_PATH): logging.error(f"FATAL: Model file not found at {MODEL_PATH}")
else:
    try:
        model = keras.models.load_model(MODEL_PATH)
        logging.info("Keras model loaded successfully.")
    except Exception as e: logging.error(f"FATAL: Error loading Keras model: {e}", exc_info=True)


# --- Initialize EasyOCR Reader (Load once) ---
ocr_reader = None
try:
    logging.info("Initializing EasyOCR Reader for CPU...")
    ocr_reader = easyocr.Reader(['en'], gpu=False)
    logging.info("EasyOCR Reader initialized.")
except Exception as e:
     logging.warning(f"Could not initialize EasyOCR reader: {e}. Live OCR will fail.", exc_info=True)
# =============================================================================


# =============================================================================
#                   --- Preprocessing & Helper Functions ---
# =============================================================================
# (Keep preprocess_image_for_model and preprocess_text_for_model)
def preprocess_image_for_model(image_path):
    # (Same as before)
    try:
        img = cv2.imread(image_path); img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img = img.astype(np.float32) / 255.0; img = (img - IMAGENET_MEAN) / IMAGENET_STD
        return np.expand_dims(img, axis=0)
    except Exception as e: logging.error(f"Error model image prep: {e}", exc_info=True); return None

def preprocess_text_for_model(text, char_map_dict):
    # (Same as before)
     if not char_map_dict: logging.error("Char_map unavailable."); return None
     if not isinstance(text, str): text = ""
     processed_text = text.strip().upper()[:MAX_TEXT_LENGTH]
     encoded = [char_map_dict.get(char, 0) for char in processed_text]
     padded = pad_sequences([encoded], maxlen=MAX_TEXT_LENGTH, padding='post', truncating='post')
     return tf.cast(padded, dtype=tf.int32)

def get_bounding_box(points):
    """Calculates the min/max x/y coordinates from EasyOCR points."""
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    xmin = math.floor(min(x_coords))
    xmax = math.ceil(max(x_coords))
    ymin = math.floor(min(y_coords))
    ymax = math.ceil(max(y_coords))
    return xmin, ymin, xmax, ymax

# =============================================================================
#                           --- Flask Routes ---
# =============================================================================
# (Keep allowed_file, index, uploaded_file routes)
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles image upload, targeted OCR, prediction, and renders result."""
    if model is None or not char_map:
        return render_template('result.html', error_message="Model/Char Map Not Loaded")

    if 'file' not in request.files: return redirect(request.url)
    file = request.files['file']
    if file.filename == '': return redirect(request.url)

    if file and allowed_file(file.filename):
        start_time = time.time()
        filename = secure_filename(file.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(image_path)
            logging.info(f"File saved to: {image_path}")
        except Exception as e:
             logging.error(f"Error saving file: {e}", exc_info=True)
             return render_template('result.html', error_message="Failed to save file.")

        # --- Perform Inference Steps ---
        # 1. Preprocess Image for the MODEL
        logging.info("Preprocessing image for model...")
        preprocessed_image_model = preprocess_image_for_model(image_path)
        if preprocessed_image_model is None:
            return render_template('result.html', error_message="Failed image prep (model).", image_filename=filename)

        # 2. Targeted OCR
        logging.info("Performing Targeted OCR using EasyOCR...")
        ocr_text = ""
        text_fragments = []
        if ocr_reader:
            try:
                # Read the original image (or grayscale version) for box detection & cropping
                img_for_ocr = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # Use grayscale
                if img_for_ocr is None: raise ValueError("Could not read image for OCR")
                img_h, img_w = img_for_ocr.shape[:2]

                # --- Step A: Detect potential text regions ---
                # detail=1 returns coordinates, text, confidence
                logging.info(" OCR - Detecting text regions...")
                initial_results = ocr_reader.readtext(img_for_ocr, detail=1, paragraph=False)
                logging.info(f" OCR - Initial regions found: {len(initial_results)}")

                # --- Step B: Crop and OCR each region ---
                for (bbox, text_guess, prob) in initial_results:
                    try:
                        # Get bounding box coordinates
                        xmin, ymin, xmax, ymax = get_bounding_box(bbox)
                        box_w = xmax - xmin
                        box_h = ymax - ymin

                        # Filter out tiny boxes (likely noise)
                        if box_w * box_h < MIN_BOX_AREA:
                            continue

                        # Add padding to crop (ensure within bounds)
                        ymin_pad = max(0, ymin - CROP_PADDING)
                        ymax_pad = min(img_h, ymax + CROP_PADDING)
                        xmin_pad = max(0, xmin - CROP_PADDING)
                        xmax_pad = min(img_w, xmax + CROP_PADDING)

                        # Crop the region from the grayscale image
                        cropped_img = img_for_ocr[ymin_pad:ymax_pad, xmin_pad:xmax_pad]

                        # Skip if crop is invalidly small after padding/clipping
                        if cropped_img.shape[0] < 5 or cropped_img.shape[1] < 5:
                            continue

                        # Optional: Apply specific preprocessing to the crop here (e.g., thresholding)
                        # _, cropped_img_thresh = cv2.threshold(cropped_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                        # --- Step C: Run OCR on the cropped region ---
                        # detail=0 just returns list of strings
                        crop_results = ocr_reader.readtext(cropped_img, detail=0, paragraph=True) # Try paragraph=True on crops

                        if crop_results:
                            text_fragments.extend(crop_results) # Add detected text fragments

                    except Exception as crop_err:
                        logging.warning(f"Error processing OCR crop: {crop_err}", exc_info=False) # Log less verbosely for loop errors
                        continue # Skip to next box on error

                ocr_text = " ".join(text_fragments).strip() # Combine text from all valid crops
                logging.info(f"OCR Result (Targeted): '{ocr_text}'")

            except Exception as e:
                logging.warning(f"Error during Targeted OCR process: {e}. Proceeding with empty text.", exc_info=True)
        else:
            logging.warning("OCR Reader not available, proceeding with empty text.")


        # 3. Preprocess Text for the MODEL
        logging.info("Preprocessing text for model...")
        preprocessed_text_model = preprocess_text_for_model(ocr_text, char_map)
        if preprocessed_text_model is None:
             return render_template('result.html', error_message="Failed text prep (model).", image_filename=filename)

        # 4. Predict
        # (Prediction logic remains the same)
        logging.info("Running model prediction...")
        try:
            prediction = model.predict([preprocessed_image_model, preprocessed_text_model], verbose=0)
            # ... (Rest of prediction and result rendering - same as before) ...
            prediction_vector = prediction[0]; predicted_class_index = np.argmax(prediction_vector)
            predicted_class_name = CLASS_NAMES[predicted_class_index]; confidence = prediction_vector[predicted_class_index] * 100
            end_time = time.time(); inference_time = end_time - start_time
            logging.info(f"Prediction: {predicted_class_name}, Confidence: {confidence:.2f}%")
            return render_template('result.html',
                                   prediction_class=predicted_class_name, confidence=confidence,
                                   image_filename=filename, ocr_text=ocr_text, # Show combined text
                                   prediction_raw=prediction_vector.tolist(), inference_time=inference_time)

        except Exception as e:
            logging.error(f"Error during model prediction: {e}", exc_info=True)
            return render_template('result.html', error_message="Prediction Error.", image_filename=filename)

    else:
        logging.warning(f"File type not allowed: {file.filename}")
        return redirect(request.url)

# (Keep /uploads/<filename> route)
@app.route('/uploads/<filename>')
def uploaded_file(filename):
     safe_filename = secure_filename(filename)
     try: return send_from_directory(app.config['UPLOAD_FOLDER'], safe_filename)
     except FileNotFoundError: logging.error(f"Upload not found: {safe_filename}"); from flask import abort; abort(404)

# (Keep __main__ block)
if __name__ == '__main__':
     if model is None or not char_map: print("\nERROR: Model/Char Map failed load.")
     else:
        print("\n--- Starting Flask Server (Targeted EasyOCR) ---"); print(f"Model: {MODEL_FILENAME}")
        print(f"Character Map Size: {len(char_map)}"); print(f"Upload Folder: {app.config['UPLOAD_FOLDER']}")
        print("Navigate to http://127.0.0.1:5000/"); app.run(host='0.0.0.0', port=5000, debug=False)