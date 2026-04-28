# -*- coding: utf-8 -*-
"""
app_optimized_threshold.py

Flask app using Tesseract OCR with enhanced image preprocessing
and an optimized prediction threshold based on validation set tuning.
Loads the model trained with class weights only. Runs on CPU.
"""

# --- Force CPU Usage for TensorFlow ---
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# -------------------------------------

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers # Ensure layers is imported globally
import cv2
import numpy as np
import pandas as pd
import pytesseract # Use pytesseract
from PIL import Image # Use Pillow for pytesseract
import time
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import logging
import math
import traceback # For detailed error logging

# =============================================================================
#                           --- Configuration ---
# =============================================================================
logging.basicConfig(level=logging.INFO)

# --- Model & Data Constants ---
IMG_WIDTH=224; IMG_HEIGHT=224; IMG_CHANNELS=3; MAX_TEXT_LENGTH=50
CLASS_NAMES = ['Counterfeit', 'Authentic'] # 0, 1
IMAGENET_MEAN=np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD=np.array([0.229, 0.224, 0.225], dtype=np.float32)

# --- Paths ---
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.join(APP_ROOT, "Caro_Laptop_Files")
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'uploads')

# !!! USER ACTION: Point to the WEIGHTED-ONLY model !!!
# Choose the correct extension (.keras or .h5) based on how it was saved
MODEL_FILENAME = r"mm_cmds_model_weighted_final.keras" # Or .keras
MODEL_PATH = os.path.join(BASE_PATH, MODEL_FILENAME)

TRAIN_DIR = os.path.join(BASE_PATH, "train")
TRAIN_CSV_OCR = os.path.join(TRAIN_DIR, "_classes_with_ocr.csv")

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Tesseract Configuration ---
# !!! USER ACTION: Uncomment and set path if needed (Windows) !!!
try:
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    logging.info(f"Explicitly set Tesseract path: {pytesseract.pytesseract.tesseract_cmd}")
    if not os.path.exists(pytesseract.pytesseract.tesseract_cmd):
         logging.warning("Tesseract executable not found at explicit path.")
except Exception:
     logging.warning("Could not set Tesseract command path. Ensure Tesseract is in system PATH.")

# --- OCR Preprocessing Config ---
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)
THRESHOLD_BLOCK_SIZE = 11 # Must be odd
THRESHOLD_C = 5

# --- Prediction Threshold ---
# !!! USER ACTION: Set the threshold determined from tune_threshold.py !!!
# Based on previous results, 0.75 looked promising. Adjust if needed.
OPTIMIZED_THRESHOLD_COUNTERFEIT = 0.75
# =============================================================================

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
if os.path.exists(TRAIN_CSV_OCR):
    try:
        # ... (char_map reconstruction code) ...
        df_train = pd.read_csv(TRAIN_CSV_OCR); df_train['extracted_text'] = df_train['extracted_text'].fillna('')
        train_texts = df_train['extracted_text'].astype(str).tolist(); all_train_text = "".join(train_texts)
        unique_chars = sorted(list(set(all_train_text))); char_map = {char: i+1 for i, char in enumerate(unique_chars)}
        text_vocab_size = len(char_map) + 1; logging.info(f"Character map OK. Vocab size: {text_vocab_size}")
    except Exception as e: logging.error(f"Error char_map: {e}", exc_info=True)
else: logging.error(f"Training CSV ({TRAIN_CSV_OCR}) not found.")

# --- Load Keras Model ---
model = None
logging.info(f"--- Loading Keras Model from {MODEL_PATH} ---")
if not os.path.exists(MODEL_PATH): logging.error(f"FATAL: Model file not found: {MODEL_PATH}")
else:
    try:
        model = keras.models.load_model(MODEL_PATH)
        logging.info(f"Keras model '{MODEL_FILENAME}' loaded successfully.")
    except Exception as e: logging.error(f"FATAL: Error loading Keras model: {e}", exc_info=True)
# =============================================================================


# =============================================================================
#                   --- Preprocessing Functions ---
# =============================================================================
# (Keep preprocess_image_for_model, preprocess_image_for_ocr_enhanced, preprocess_text_for_model)
def preprocess_image_for_model(image_path):
    try:
        img = cv2.imread(image_path); img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img = img.astype(np.float32) / 255.0; img = (img - IMAGENET_MEAN) / IMAGENET_STD
        return np.expand_dims(img, axis=0)
    except Exception as e: logging.error(f"Error model image prep: {e}"); return None

def preprocess_image_for_ocr_enhanced(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None: logging.warning(f"OCR Prep: Cannot load {image_path}"); return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
        contrast_enhanced = clahe.apply(gray)
        binary_img = cv2.adaptiveThreshold(contrast_enhanced, 255,
                                           cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                           THRESHOLD_BLOCK_SIZE, THRESHOLD_C)
        img_pil = Image.fromarray(binary_img)
        logging.info("Applied Grayscale, CLAHE, Adaptive Thresholding for OCR.")
        return img_pil
    except Exception as e: logging.error(f"Error OCR image prep: {e}"); return None

def preprocess_text_for_model(text, char_map_dict):
     if not char_map_dict: logging.error("Char_map unavailable."); return None
     if not isinstance(text, str): text = ""
     processed_text = text.strip().upper()[:MAX_TEXT_LENGTH]
     encoded = [char_map_dict.get(char, 0) for char in processed_text]
     padded = pad_sequences([encoded], maxlen=MAX_TEXT_LENGTH, padding='post', truncating='post')
     return tf.cast(padded, dtype=tf.int32)
# =============================================================================


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
    """Handles upload, preprocessing, OCR, prediction with threshold."""
    if model is None or not char_map: return render_template('result.html', error_message="Model/Char Map Not Loaded")
    if 'file' not in request.files: return redirect(request.url)
    file = request.files['file']
    if file.filename == '': return redirect(request.url)

    if file and allowed_file(file.filename):
        start_time = time.time()
        filename = secure_filename(file.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try: file.save(image_path); logging.info(f"File saved: {image_path}")
        except Exception as e: logging.error(f"Error saving: {e}"); return render_template('result.html', error_message="Save Error.")

        # 1. Preprocess Image for MODEL
        logging.info("Preprocessing image for model...")
        preprocessed_image_model = preprocess_image_for_model(image_path)
        if preprocessed_image_model is None: return render_template('result.html', error_message="Model Img Prep Error.", image_filename=filename)

        # 2. Preprocess Image for OCR & Run Tesseract
        logging.info("Preprocessing image for Tesseract OCR...")
        img_pil_for_ocr = preprocess_image_for_ocr_enhanced(image_path)
        ocr_text = ""
        if img_pil_for_ocr is not None:
            logging.info("Performing OCR using Tesseract...")
            try:
                tesseract_config = '--psm 6' # Default: Assume single block
                ocr_text = pytesseract.image_to_string(img_pil_for_ocr, lang='eng', config=tesseract_config)
                ocr_text = ocr_text.replace('\n', ' ').replace('\f', '').strip()
                logging.info(f"Tesseract OCR Result (Config: {tesseract_config}): '{ocr_text}'")
            except Exception as e: logging.warning(f"Error during Tesseract OCR: {e}", exc_info=True) # Log full trace on warning
        else: logging.warning("Image preprocessing for OCR failed.")

        # 3. Preprocess Text for MODEL
        logging.info("Preprocessing text for model...")
        preprocessed_text_model = preprocess_text_for_model(ocr_text, char_map)
        if preprocessed_text_model is None: return render_template('result.html', error_message="Model Text Prep Error.", image_filename=filename)

        # 4. Predict
        logging.info("Running model prediction...");
        try:
            prediction = model.predict([preprocessed_image_model, preprocessed_text_model], verbose=0)
            prediction_vector = prediction[0] # Probabilities [C, A]

            # *** APPLY OPTIMIZED THRESHOLD ***
            logging.info(f"Applying threshold: {OPTIMIZED_THRESHOLD_COUNTERFEIT} on Counterfeit probability ({prediction_vector[0]:.4f})")
            if prediction_vector[0] >= OPTIMIZED_THRESHOLD_COUNTERFEIT:
                predicted_class_index = 0 # Counterfeit
            else:
                predicted_class_index = 1 # Authentic
            # *********************************

            predicted_class_name = CLASS_NAMES[predicted_class_index]
            confidence = prediction_vector[predicted_class_index] * 100 # Confidence in the *predicted* class
            end_time = time.time(); inference_time = end_time - start_time
            logging.info(f"Prediction: {predicted_class_name}, Confidence: {confidence:.2f}%")

            # 5. Render Result
            return render_template('result.html',
                                   prediction_class=predicted_class_name, confidence=confidence,
                                   image_filename=filename, ocr_text=ocr_text,
                                   prediction_raw=prediction_vector.tolist(), inference_time=inference_time)
        except Exception as e: logging.error(f"Prediction Error: {e}", exc_info=True); return render_template('result.html', error_message="Prediction Error.", image_filename=filename)
    else: logging.warning(f"File not allowed: {file.filename}"); return redirect(request.url)

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
        print("\n--- Starting Flask Server (Enhanced Tesseract OCR + Optimized Threshold) ---")
        print(f"Model: {MODEL_FILENAME}"); print(f"Character Map Size: {len(char_map)}")
        print(f"Upload Folder: {app.config['UPLOAD_FOLDER']}")
        print(f"Using Prediction Threshold (Counterfeit >=): {OPTIMIZED_THRESHOLD_COUNTERFEIT}")
        print("Navigate to http://127.0.0.1:5000/"); app.run(host='0.0.0.0', port=5000, debug=False)