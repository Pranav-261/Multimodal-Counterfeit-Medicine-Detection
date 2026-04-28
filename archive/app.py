# -*- coding: utf-8 -*-
"""
app.py

Flask web application for the multi-modal counterfeit medicine detection model.
Allows uploading an image, runs OCR and model prediction (on CPU),
and displays the result.
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
import easyocr
import time
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename # For secure filename handling
import logging

# =============================================================================
#                           --- Configuration ---
# =============================================================================
# Configure logging
logging.basicConfig(level=logging.INFO)

# --- Model & Data Constants ---
IMG_WIDTH = 224
IMG_HEIGHT = 224
IMG_CHANNELS = 3
MAX_TEXT_LENGTH = 50
CLASS_NAMES = ['Counterfeit', 'Authentic']
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# --- Paths ---
# Use absolute paths for Flask robustness or ensure relative paths are correct from where you run Flask
APP_ROOT = os.path.dirname(os.path.abspath(__file__)) # Gets the directory where app.py is
BASE_PATH = os.path.join(APP_ROOT, "Caro_Laptop_Files") # Assumes Caro_Laptop_Files is in the same dir as app.py
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'uploads')
MODEL_FILENAME = r"mm_cmds_model_finetune_robust_final.h5" # Use the weighted model
MODEL_PATH = os.path.join(BASE_PATH, MODEL_FILENAME)
TRAIN_DIR = os.path.join(BASE_PATH, "train")
TRAIN_CSV_OCR = os.path.join(TRAIN_DIR, "_classes_with_ocr.csv")

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =============================================================================
#            --- Initialize App, Load Model & Char Map ---
# =============================================================================

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # Limit upload size (e.g., 16MB)

# --- Load Character Map (Reconstruct or load from file) ---
char_map = {}
text_vocab_size = 1
logging.info("--- Reconstructing Character Map ---")
if os.path.exists(TRAIN_CSV_OCR):
    try:
        df_train = pd.read_csv(TRAIN_CSV_OCR)
        df_train['extracted_text'] = df_train['extracted_text'].fillna('')
        train_texts = df_train['extracted_text'].astype(str).tolist()
        all_train_text = "".join(train_texts)
        unique_chars = sorted(list(set(all_train_text)))
        char_map = {char: i+1 for i, char in enumerate(unique_chars)}
        text_vocab_size = len(char_map) + 1
        logging.info(f"Character map reconstructed. Vocab size: {text_vocab_size}")
    except Exception as e:
        logging.error(f"Error reconstructing char_map from {TRAIN_CSV_OCR}: {e}", exc_info=True)
        # Decide how critical this is - maybe app can't start?
else:
    logging.error(f"Error: Training CSV with OCR ({TRAIN_CSV_OCR}) not found. Cannot reconstruct char_map.")
    # Consider exiting or running without text features if char_map fails

# --- Load Keras Model ---
model = None
logging.info(f"--- Loading Keras Model from {MODEL_PATH} ---")
if not os.path.exists(MODEL_PATH):
    logging.error(f"FATAL: Model file not found at {MODEL_PATH}")
    # App probably shouldn't run without the model
else:
    try:
        model = keras.models.load_model(MODEL_PATH)
        logging.info("Keras model loaded successfully.")
        # Perform a dummy prediction to "warm up" the model if needed
        # dummy_img = np.zeros((1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)
        # dummy_txt = tf.zeros((1, MAX_TEXT_LENGTH), dtype=tf.int32)
        # model.predict([dummy_img, dummy_txt], verbose=0)
        # logging.info("Model warm-up prediction complete.")
    except Exception as e:
        logging.error(f"FATAL: Error loading Keras model: {e}", exc_info=True)
        # App probably shouldn't run without the model

# --- Initialize EasyOCR Reader (Load once) ---
# Load it for CPU explicitly as TF is forced to CPU
ocr_reader = None
try:
    logging.info("Initializing EasyOCR Reader for CPU...")
    ocr_reader = easyocr.Reader(['en'], gpu=False)
    logging.info("EasyOCR Reader initialized.")
except Exception as e:
     logging.warning(f"Could not initialize EasyOCR reader: {e}. Live OCR will fail.", exc_info=True)


# =============================================================================
#                   --- Preprocessing Functions ---
# =============================================================================
# (Identical to the ones in predict_medicine.py)
def preprocess_image_for_inference(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None: return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img = img.astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        return np.expand_dims(img, axis=0)
    except Exception as e:
        logging.error(f"Error processing image {image_path}: {e}", exc_info=True)
        return None

def preprocess_text_for_inference(text, char_map_dict):
    if not char_map_dict: logging.error("Char_map unavailable for text encoding."); return None
    if not isinstance(text, str): text = ""
    processed_text = text.strip().upper()[:MAX_TEXT_LENGTH]
    encoded = [char_map_dict.get(char, 0) for char in processed_text]
    padded = pad_sequences([encoded], maxlen=MAX_TEXT_LENGTH, padding='post', truncating='post')
    return tf.cast(padded, dtype=tf.int32)

# =============================================================================
#                           --- Flask Routes ---
# =============================================================================

def allowed_file(filename):
    """Checks if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def index():
    """Renders the main upload page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles image upload, preprocessing, prediction, and renders result."""
    if model is None or not char_map:
        return render_template('result.html', error_message="Model or Character Map not loaded properly. Check server logs.")

    # Check if the post request has the file part
    if 'file' not in request.files:
        logging.warning("No file part in request.")
        return redirect(request.url)
    file = request.files['file']

    # If the user does not select a file, the browser submits an empty file without a filename.
    if file.filename == '':
        logging.warning("No selected file.")
        return redirect(request.url)

    if file and allowed_file(file.filename):
        start_time = time.time()
        filename = secure_filename(file.filename) # Sanitize filename
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(image_path)
            logging.info(f"File saved to: {image_path}")
        except Exception as e:
             logging.error(f"Error saving uploaded file: {e}", exc_info=True)
             return render_template('result.html', error_message="Failed to save uploaded file.")

        # 1. Preprocess Image
        logging.info("Preprocessing image...")
        preprocessed_image = preprocess_image_for_inference(image_path)
        if preprocessed_image is None:
            return render_template('result.html', error_message="Failed to preprocess image.", image_filename=filename)

        # 2. Run Live OCR
        logging.info("Performing OCR...")
        ocr_text = ""
        if ocr_reader:
            try:
                results = ocr_reader.readtext(image_path)
                ocr_text = " ".join([res[1] for res in results if res])
                logging.info(f"OCR Result: '{ocr_text}'")
            except Exception as e:
                logging.warning(f"Error during live OCR: {e}. Proceeding with empty text.", exc_info=True)
        else:
            logging.warning("OCR Reader not available, proceeding with empty text.")


        # 3. Preprocess Text
        logging.info("Preprocessing text...")
        preprocessed_text = preprocess_text_for_inference(ocr_text, char_map)
        if preprocessed_text is None:
             return render_template('result.html', error_message="Failed to preprocess text.", image_filename=filename)

        # 4. Predict
        logging.info("Running model prediction...")
        try:
            prediction = model.predict([preprocessed_image, preprocessed_text], verbose=0)
            prediction_vector = prediction[0]
            predicted_class_index = np.argmax(prediction_vector)
            predicted_class_name = CLASS_NAMES[predicted_class_index]
            confidence = prediction_vector[predicted_class_index] * 100
            end_time = time.time()
            inference_time = end_time - start_time
            logging.info(f"Prediction: {predicted_class_name}, Confidence: {confidence:.2f}%")

            # 5. Render Result
            return render_template('result.html',
                                   prediction_class=predicted_class_name,
                                   confidence=confidence,
                                   image_filename=filename,
                                   ocr_text=ocr_text,
                                   prediction_raw=prediction_vector.tolist(), 
                                   inference_time=inference_time)

        except Exception as e:
            logging.error(f"Error during model prediction: {e}", exc_info=True)
            return render_template('result.html', error_message="Error during model prediction.", image_filename=filename)

    else:
        logging.warning(f"File type not allowed: {file.filename}")
        return redirect(request.url) # Or show an error message

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serves files from the upload folder."""
    safe_filename = secure_filename(filename)
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], safe_filename)
    except FileNotFoundError:
        logging.error(f"Attempted to access non-existent file in uploads: {safe_filename}")
        from flask import abort
        abort(404)

if __name__ == '__main__':
    if model is None or not char_map:
         print("\nERROR: Model or Character Map failed to load. Flask app will not start.")
         print("Please check the file paths and logs above.")
    else:
        print("\n--- Starting Flask Development Server ---")
        print(f"Model: {MODEL_FILENAME}")
        print(f"Character Map Size: {len(char_map)}")
        print(f"Upload Folder: {app.config['UPLOAD_FOLDER']}")
        print("Navigate to http://127.0.0.1:5000/ in your browser.")
        app.run(host='0.0.0.0', port=5000, debug=False)