# -*- coding: utf-8 -*-
"""
app_paddle_ocr_robust_parsing.py
** current **

Flask app using PaddleOCR with more robust result parsing.
Loads the fine-tuned model. Applies optimized threshold. Runs on CPU.
"""

# --- Force CPU Usage for TensorFlow ---
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# -------------------------------------

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers
import cv2
import numpy as np
import pandas as pd
from paddleocr import PaddleOCR # Import PaddleOCR
import time
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import logging
import math
import traceback

# =============================================================================
#                           --- Configuration ---
# =============================================================================
logging.basicConfig(level=logging.INFO)
IMG_WIDTH=224; IMG_HEIGHT=224; IMG_CHANNELS=3; MAX_TEXT_LENGTH=50
CLASS_NAMES = ['Counterfeit', 'Authentic']
IMAGENET_MEAN=np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD=np.array([0.229, 0.224, 0.225], dtype=np.float32)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.join(APP_ROOT, "Caro_Laptop_Files_CROPPED")  # Use the cropped dataset
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'uploads')
MODEL_FILENAME = r"mm_cmds_model_paddle_ft_robust_v4.h5"  # Use the final model with PaddleOCR
MODEL_PATH = os.path.join(BASE_PATH, MODEL_FILENAME)
TRAIN_DIR = os.path.join(BASE_PATH, "train")
TRAIN_CSV_OCR = os.path.join(TRAIN_DIR, "_classes_with_paddle_ocr.csv")
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
OPTIMIZED_THRESHOLD_COUNTERFEIT = 0.40
# =============================================================================

# =============================================================================
#            --- Initialize App, Load Model & Char Map, PaddleOCR ---
# =============================================================================
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

char_map = {}; text_vocab_size = 1
logging.info("--- Reconstructing Character Map ---")
if os.path.exists(TRAIN_CSV_OCR):
    try:
        df_train = pd.read_csv(TRAIN_CSV_OCR); df_train['extracted_text'] = df_train['extracted_text'].fillna('')
        train_texts = df_train['extracted_text'].astype(str).tolist(); all_train_text = "".join(train_texts)
        unique_chars = sorted(list(set(all_train_text))); char_map = {char: i+1 for i, char in enumerate(unique_chars)}
        text_vocab_size = len(char_map) + 1; logging.info(f"Character map OK. Vocab size: {text_vocab_size}")
    except Exception as e: logging.error(f"Error char_map: {e}", exc_info=True)
else: logging.error(f"Training CSV ({TRAIN_CSV_OCR}) not found for char_map.")

model = None
logging.info(f"--- Loading Keras Model from {MODEL_PATH} ---")
if not os.path.exists(MODEL_PATH): logging.error(f"FATAL: Model file not found: {MODEL_PATH}")
else:
    try: model = keras.models.load_model(MODEL_PATH); logging.info(f"Keras model '{MODEL_FILENAME}' loaded.")
    except Exception as e: logging.error(f"FATAL: Error loading Keras model: {e}", exc_info=True)

paddle_ocr_reader = None
try:
    logging.info("Initializing PaddleOCR Reader (CPU)...")
    paddle_ocr_reader = PaddleOCR(use_angle_cls=True, lang='en')
    logging.info("PaddleOCR Reader initialized successfully.")
except Exception as e:
     logging.warning(f"Could not initialize PaddleOCR reader: {e}. Live OCR will fail.", exc_info=True)
     paddle_ocr_reader = None
# =============================================================================

# =============================================================================
#                   --- Preprocessing Functions ---
# =============================================================================
def preprocess_image_for_model(image_path):
    try: img = cv2.imread(image_path); img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB); img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT)); img = img.astype(np.float32) / 255.0; img = (img - IMAGENET_MEAN) / IMAGENET_STD; return np.expand_dims(img, axis=0)
    except Exception as e: logging.error(f"Error model image prep: {e}"); return None

def preprocess_image_for_paddle_ocr(image_path):
    try:
        if not os.path.exists(image_path): logging.warning(f"PaddleOCR Prep: Image path does not exist: {image_path}"); return None
        return image_path # Pass the path, PaddleOCR can handle it
    except Exception as e: logging.error(f"Error OCR image prep: {e}"); return None

def preprocess_text_for_model(text, char_map_dict):
     if not char_map_dict: logging.error("Char_map unavailable."); return None
     if not isinstance(text, str): text = ""
     processed_text = text.strip().upper()[:MAX_TEXT_LENGTH]; encoded = [char_map_dict.get(char, 0) for char in processed_text]
     padded = pad_sequences([encoded], maxlen=MAX_TEXT_LENGTH, padding='post', truncating='post'); return tf.cast(padded, dtype=tf.int32)
# =============================================================================

# =============================================================================
#                           --- Flask Routes ---
# =============================================================================
def allowed_file(filename): return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
@app.route('/', methods=['GET'])
def index(): return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or not char_map: return render_template('result.html', error_message="Model/Char Map Not Loaded")
    if 'file' not in request.files: return redirect(request.url)
    file = request.files['file'];
    if file.filename == '': return redirect(request.url)

    if file and allowed_file(file.filename):
        start_time = time.time(); filename = secure_filename(file.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try: file.save(image_path); logging.info(f"File saved: {image_path}")
        except Exception as e: logging.error(f"Error saving: {e}"); return render_template('result.html', error_message="Save Error.")

        preprocessed_image_model = preprocess_image_for_model(image_path)
        if preprocessed_image_model is None: return render_template('result.html', error_message="Model Img Prep Error.", image_filename=filename)

        logging.info("Performing OCR using PaddleOCR...")
        ocr_text = ""
        input_for_paddle = preprocess_image_for_paddle_ocr(image_path)

        if paddle_ocr_reader and input_for_paddle is not None:
            try:
                result = paddle_ocr_reader.ocr(input_for_paddle)
                logging.info(f"PADDLE_OCR_RAW_RESULT (type: {type(result)}): {result}")

                lines = []
                if result and isinstance(result, list) and len(result) > 0:
                    detections_for_image = result[0] # First element for the first (and only) image
                    if detections_for_image is not None: # Check if it's not None
                        if isinstance(detections_for_image, list): # Standard case: list of lines
                            for line_info_item in detections_for_image:
                                if line_info_item and len(line_info_item) == 2 and \
                                   isinstance(line_info_item[1], tuple) and len(line_info_item[1]) >= 1:
                                    lines.append(line_info_item[1][0])
                                else:
                                    logging.warning(f"Unexpected line_info_item format: {line_info_item}")
                        elif isinstance(detections_for_image, dict) and 'rec_texts' in detections_for_image:
                            logging.info("PaddleOCR result[0] is a dictionary, trying 'rec_texts'.")
                            if isinstance(detections_for_image['rec_texts'], list):
                                lines.extend(detections_for_image['rec_texts'])
                            else:
                                logging.warning(f"'rec_texts' not a list: {type(detections_for_image['rec_texts'])}")
                        else:
                            logging.warning(f"Unexpected structure for detections_for_image: {type(detections_for_image)}")
                    else:
                        logging.info("PaddleOCR result[0] (detections_for_image) is None.")
                else:
                    logging.info("PaddleOCR returned None or empty result list.")

                ocr_text = " ".join(lines).strip()
                logging.info(f"PaddleOCR Processed Text: '{ocr_text}'")
            except Exception as e:
                logging.warning(f"Error during PaddleOCR processing: {e}. Proceeding with empty text.", exc_info=True)
        elif not paddle_ocr_reader: logging.error("PaddleOCR reader not initialized.")
        else: logging.warning("Input for PaddleOCR is invalid (path or data).")

        preprocessed_text_model = preprocess_text_for_model(ocr_text, char_map)
        if preprocessed_text_model is None: return render_template('result.html', error_message="Model Text Prep Error.", image_filename=filename)

        logging.info("Running model prediction...");
        try:
            prediction = model.predict([preprocessed_image_model, preprocessed_text_model], verbose=0)
            prediction_vector = prediction[0]
            logging.info(f"Applying threshold: {OPTIMIZED_THRESHOLD_COUNTERFEIT} on P(C)={prediction_vector[0]:.4f}")
            if prediction_vector[0] >= OPTIMIZED_THRESHOLD_COUNTERFEIT: predicted_class_index = 0
            else: predicted_class_index = 1
            predicted_class_name = CLASS_NAMES[predicted_class_index]
            confidence = prediction_vector[predicted_class_index] * 100
            end_time = time.time(); inference_time = end_time - start_time
            logging.info(f"Prediction: {predicted_class_name}, Confidence: {confidence:.2f}%")
            return render_template('result.html', prediction_class=predicted_class_name, confidence=confidence,
                                   image_filename=filename, ocr_text=ocr_text,
                                   prediction_raw=prediction_vector.tolist(), inference_time=inference_time)
        except Exception as e: logging.error(f"Prediction Error: {e}", exc_info=True); return render_template('result.html', error_message="Prediction Error.", image_filename=filename)
    else: logging.warning(f"File not allowed: {file.filename}"); return redirect(request.url)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
     safe_filename = secure_filename(filename)
     try: return send_from_directory(app.config['UPLOAD_FOLDER'], safe_filename)
     except FileNotFoundError: logging.error(f"Upload not found: {safe_filename}"); from flask import abort; abort(404)

if __name__ == '__main__':
     if model is None or not char_map: print("\nERROR: Model/Char Map failed load.")
     elif paddle_ocr_reader is None: print("\nERROR: PaddleOCR Reader failed to initialize. App cannot start.")
     else:
        print("\n--- Starting Flask Server (PaddleOCR Corrected Parsing + Optimized Threshold) ---")
        print(f"Using Model: {MODEL_FILENAME}"); print(f"Character Map Size: {len(char_map)}")
        print(f"Using Prediction Threshold (Counterfeit if P(C) >=): {OPTIMIZED_THRESHOLD_COUNTERFEIT}")
        print("Navigate to http://127.0.0.1:5000/"); app.run(host='0.0.0.0', port=5000, debug=False)