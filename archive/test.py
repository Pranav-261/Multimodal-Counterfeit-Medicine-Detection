# -*- coding: utf-8 -*-
"""
predict_medicine_ocr_preprocessed.py

Loads the trained model and performs inference, adding image
preprocessing steps specifically to improve live OCR results.
Runs TensorFlow on CPU.

Requirements:
  - tensorflow
  - opencv-python-headless
  - easyocr
  - pandas (only if reconstructing char_map)
  - numpy

Usage:
  1. Update constants/paths.
  2. Ensure Option B (Live OCR) is selected.
  3. Run from terminal: python predict_medicine_ocr_preprocessed.py
"""

# --- Force CPU Usage for TensorFlow ---
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# -------------------------------------

import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
import easyocr
import time
import logging # Use logging for better messages

logging.basicConfig(level=logging.INFO)

# =============================================================================
#                           --- Configuration ---
# =============================================================================
# (Keep Constants and Paths sections the same, update BASE_PATH etc. as needed)
IMG_WIDTH = 224; IMG_HEIGHT = 224; IMG_CHANNELS = 3; MAX_TEXT_LENGTH = 50
CLASS_NAMES = ['Counterfeit', 'Authentic']
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
BASE_PATH = r"Caro_Laptop_Files"
MODEL_FILENAME = r"mm_cmds_model_weighted_final.keras" # Use the weighted model
MODEL_PATH = os.path.join(BASE_PATH, MODEL_FILENAME)
TEST_IMAGE_PATH = os.path.join(BASE_PATH, "test", "Pick-a-style-thats-simple-to-open-4-1.jpg") # CHANGE THIS (Using Camfed image)
TRAIN_DIR = os.path.join(BASE_PATH, "train")
TRAIN_CSV_OCR = os.path.join(TRAIN_DIR, "_classes_with_ocr.csv")

# --- OCR Preprocessing Config ---
OCR_UPSCALE_FACTOR = 1.5 # How much to enlarge image before OCR (1.0 = no upscale)
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)
# =============================================================================

# =============================================================================
#        --- Character Map Loading / Reconstruction ---
# =============================================================================
logging.info("--- Reconstructing Character Map ---")
char_map = {}
text_vocab_size = 1
# ... (char_map reconstruction code - same as before) ...
if os.path.exists(TRAIN_CSV_OCR):
    try:
        df_train = pd.read_csv(TRAIN_CSV_OCR)
        df_train['extracted_text'] = df_train['extracted_text'].fillna('')
        train_texts = df_train['extracted_text'].astype(str).tolist()
        all_train_text = "".join(train_texts); unique_chars = sorted(list(set(all_train_text)))
        char_map = {char: i+1 for i, char in enumerate(unique_chars)}; text_vocab_size = len(char_map) + 1
        logging.info(f"Character map reconstructed. Vocab size: {text_vocab_size}")
    except Exception as e: logging.error(f"Error reconstructing char_map: {e}", exc_info=True); exit()
else: logging.error(f"Training CSV ({TRAIN_CSV_OCR}) not found."); exit()
# =============================================================================

# =============================================================================
#                   --- Preprocessing Functions ---
# =============================================================================

# --- Preprocessing FOR THE MODEL (Input to Keras Model) ---
def preprocess_image_for_model(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None: logging.warning(f"Failed model preprocessing: load {image_path}"); return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT)) # Resize to model input size
        img = img.astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        return np.expand_dims(img, axis=0)
    except Exception as e:
        logging.error(f"Error processing image {image_path} for model: {e}", exc_info=True)
        return None

# --- Preprocessing FOR OCR (Input to EasyOCR) ---
def preprocess_image_for_ocr(image_path):
    """Applies preprocessing to enhance text for OCR."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            logging.warning(f"Failed OCR preprocessing: load {image_path}")
            return None

        # 1. Convert to Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
        enhanced_contrast = clahe.apply(gray)

        # 3. Upscale (Optional)
        if OCR_UPSCALE_FACTOR > 1.0:
            new_width = int(enhanced_contrast.shape[1] * OCR_UPSCALE_FACTOR)
            new_height = int(enhanced_contrast.shape[0] * OCR_UPSCALE_FACTOR)
            upscaled = cv2.resize(enhanced_contrast, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            logging.info(f"Image upscaled for OCR to {upscaled.shape}")
            return upscaled
        else:
            return enhanced_contrast # Return contrast-enhanced grayscale if no upscaling

    except Exception as e:
        logging.error(f"Error during image preprocessing for OCR {image_path}: {e}", exc_info=True)
        return None

# --- Text Preprocessing (Input to Keras Model) ---
def preprocess_text_for_model(text, char_map_dict):
    if not char_map_dict: logging.error("Char_map unavailable."); return None
    if not isinstance(text, str): text = ""
    processed_text = text.strip().upper()[:MAX_TEXT_LENGTH]
    encoded = [char_map_dict.get(char, 0) for char in processed_text]
    padded = pad_sequences([encoded], maxlen=MAX_TEXT_LENGTH, padding='post', truncating='post')
    return tf.cast(padded, dtype=tf.int32)

# =============================================================================
#                           --- Main Inference ---
# =============================================================================
def main():
    logging.info("\n--- Checking TensorFlow Device ---")
    logging.info(f"Visible Physical Devices: {tf.config.list_physical_devices()}")

    logging.info(f"\n--- Starting Prediction (CPU Mode, Live OCR with Preprocessing) ---")
    # --- Load Model ---
    logging.info(f"Loading model from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH): logging.error("Model file not found."); return
    try:
        model = keras.models.load_model(MODEL_PATH)
        logging.info("Model loaded successfully.")
    except Exception as e: logging.error(f"Error loading Keras model: {e}", exc_info=True); return

    # --- Prepare Inputs ---
    logging.info(f"Processing image: {TEST_IMAGE_PATH}")
    if not os.path.exists(TEST_IMAGE_PATH): logging.error("Test image not found."); return

    # 1. Preprocess Image for the MODEL
    preprocessed_image_model = preprocess_image_for_model(TEST_IMAGE_PATH)
    if preprocessed_image_model is None: logging.error("Failed model image preprocessing."); return

    # 2. Preprocess Image for OCR & Run OCR
    logging.info("Preprocessing image for OCR...")
    preprocessed_image_ocr = preprocess_image_for_ocr(TEST_IMAGE_PATH)
    ocr_text = ""
    if preprocessed_image_ocr is not None:
        logging.info("Attempting live OCR on preprocessed image (CPU)...")
        try:
            ocr_reader = easyocr.Reader(['en'], gpu=False)
            logging.info("EasyOCR reader initialized for CPU.")
            # *** Feed the processed NUMPY ARRAY to readtext ***
            results = ocr_reader.readtext(preprocessed_image_ocr)
            # =================================================
            logging.info(f"DEBUG: Raw EasyOCR results after preprocessing: {results}")
            ocr_text = " ".join([res[1] for res in results if res])
            logging.info(f"OCR Extracted Text (Post-Preprocessing): '{ocr_text}'")
            del ocr_reader
        except ImportError: logging.error("EasyOCR not installed."); return
        except Exception as e: logging.warning(f"Error during live OCR: {e}. Proceeding with empty text.", exc_info=True)
    else:
        logging.warning("Image preprocessing for OCR failed. Skipping OCR.")


    # 3. Preprocess Text for the MODEL
    logging.info("Preprocessing text for model...")
    preprocessed_text_model = preprocess_text_for_model(ocr_text, char_map)
    if preprocessed_text_model is None: logging.error("Failed model text preprocessing."); return

    # --- Predict ---
    logging.info("\nModel Input Shapes:")
    logging.info(f"  Image: {preprocessed_image_model.shape}")
    logging.info(f"  Text:  {preprocessed_text_model.shape}")

    logging.info("Running model prediction (on CPU)...")
    start_pred_time = time.time()
    prediction = model.predict([preprocessed_image_model, preprocessed_text_model], verbose=0)
    end_pred_time = time.time()

    # --- Interpret and Print Results ---
    # (Result interpretation code remains the same)
    if prediction is not None and len(prediction) > 0:
        prediction_vector = prediction[0]; predicted_class_index = np.argmax(prediction_vector)
        predicted_class_name = CLASS_NAMES[predicted_class_index]; confidence = prediction_vector[predicted_class_index] * 100
        print("\n--- Prediction Result ---") # Use print for final user output
        print(f"Input Image: {os.path.basename(TEST_IMAGE_PATH)}"); print(f"Input Text (used): '{ocr_text}'")
        print(f"Raw Output Probabilities [Counterfeit, Authentic]: {prediction_vector}")
        print(f"Predicted Class: {predicted_class_name} (Index: {predicted_class_index})")
        print(f"Confidence: {confidence:.2f}%"); print(f"Inference Time: {end_pred_time - start_pred_time:.4f} seconds")
    else: print("Model prediction failed.")
# =============================================================================

if __name__ == "__main__":
    main()