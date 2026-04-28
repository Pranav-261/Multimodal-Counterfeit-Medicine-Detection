# -*- coding: utf-8 -*-
"""
analyze_prediction.py

Loads a trained model, makes a prediction on a specific image,
and generates a Grad-CAM heatmap to visualize image feature importance.
"""

# --- Force CPU Usage ---
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# -------------------------------------

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
# from tf_explain.core.grad_cam import GradCAM # Using manual Grad-CAM implementation now
import pytesseract # Or EasyOCR if you prefer for text extraction here
from PIL import Image
import logging
import traceback

logging.basicConfig(level=logging.INFO)

# =============================================================================
#                           --- Configuration ---
# =============================================================================
# --- Model & Data Constants ---
# (Copy relevant constants: IMG_WIDTH, HEIGHT, CHANNELS, MAX_TEXT_LENGTH, CLASS_NAMES, MEAN, STD)
IMG_WIDTH=224; IMG_HEIGHT=224; IMG_CHANNELS=3; MAX_TEXT_LENGTH=50
CLASS_NAMES = ['Counterfeit', 'Authentic']
IMAGENET_MEAN=np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD=np.array([0.229, 0.224, 0.225], dtype=np.float32)

# --- Paths ---
BASE_PATH = r"Caro_Laptop_Files"
# *** Choose the model you want to analyze ***
MODEL_FILENAME = r"mm_cmds_model_weighted_final.keras" # e.g., The weighted-only model
MODEL_PATH = os.path.join(BASE_PATH, MODEL_FILENAME)

# *** Choose the specific image you want to analyze ***
IMAGE_TO_ANALYZE = os.path.join(BASE_PATH, "test","191489945_4122074827848540_4141220302232318049_n_jpg.rf.b72edfea0236850b0baab874268c29e0.jpg") # e.g., The Camfed image
# Or choose a known counterfeit image path

TRAIN_DIR = os.path.join(BASE_PATH, "train") # For char_map
TRAIN_CSV_OCR = os.path.join(TRAIN_DIR, "_classes_with_ocr.csv")

# --- Grad-CAM Configuration ---
# Find this layer name using model.summary() on the loaded model
# Should be the last layer *before* GlobalAveragePooling in the image branch
LAST_CONV_LAYER_NAME = 'out_relu' # For MobileNetV2 often 'out_relu' or 'block_16_project_BN'
# Or find it programmatically if needed:
# last_conv_layer_name = ""
# for layer in reversed(model.layers):
#     if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
#          if 'input' not in layer.name.lower() and 'pad' not in layer.name.lower(): # Avoid first/padding layers
#               last_conv_layer_name = layer.name
#               logging.info(f"Auto-detected last conv layer: {last_conv_layer_name}")
#               break

# --- Tesseract Config ---
try:
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
except Exception: pass # Ignore if already in PATH
# =============================================================================


# =============================================================================
#            --- Char Map, Preprocessing Functions ---
# =============================================================================
# (Include char_map loading, preprocess_image_for_model,
#  preprocess_image_for_ocr_enhanced, preprocess_text_for_model here - same as app.py)
char_map = {}; text_vocab_size = 1
logging.info("--- Reconstructing Character Map ---")
if os.path.exists(TRAIN_CSV_OCR):
    try:
        df_train = pd.read_csv(TRAIN_CSV_OCR); df_train['extracted_text'] = df_train['extracted_text'].fillna('')
        train_texts = df_train['extracted_text'].astype(str).tolist(); all_train_text = "".join(train_texts)
        unique_chars = sorted(list(set(all_train_text))); char_map = {char: i+1 for i, char in enumerate(unique_chars)}
        text_vocab_size = len(char_map) + 1; logging.info(f"Character map OK. Vocab size: {text_vocab_size}")
    except Exception as e: logging.error(f"Error char_map: {e}")
else: logging.error(f"Training CSV ({TRAIN_CSV_OCR}) not found.")

def preprocess_image_for_model(image_path):
    try:
        img = cv2.imread(image_path); img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img = img.astype(np.float32) / 255.0; img = (img - IMAGENET_MEAN) / IMAGENET_STD
        return np.expand_dims(img, axis=0)
    except Exception as e: logging.error(f"Error model image prep: {e}"); return None

def preprocess_image_for_ocr_enhanced(image_path):
    try:
        img = cv2.imread(image_path); gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)); contrast_enhanced = clahe.apply(gray)
        binary_img = cv2.adaptiveThreshold(contrast_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5)
        return Image.fromarray(binary_img) # Return PIL Image
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
#                     --- Grad-CAM Implementation ---
# =============================================================================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, class_index, text_array):
    """Creates a Grad-CAM heatmap for a specific class index."""
    # Create a model that maps the multi-inputs to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        inputs=model.inputs, # Takes both image and text inputs
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Compute gradients with respect to the specific class prediction
    with tf.GradientTape() as tape:
        # Run inputs through the grad_model
        last_conv_layer_output, preds = grad_model([img_array, text_array])
        # Get the score for the target class
        if class_index is None: # If no index provided, use the predicted class
            class_index = tf.argmax(preds[0])
        class_channel = preds[:, class_index]

    # This is the gradient of the output neuron (for the selected class)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)
    if grads is None:
         logging.error("GradCAM: Gradient calculation failed. Check layer names or model structure.")
         return None

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)) # Pool spatial dimensions

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the target class
    last_conv_layer_output = last_conv_layer_output[0] # Remove batch dim
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap) # Remove channel dim

    # For visualization, we normalize the heatmap between 0 & 1 and apply ReLU
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + tf.keras.backend.epsilon())
    return heatmap.numpy(), class_index # Return heatmap and the index it explained

def display_gradcam(img_path, heatmap, class_name, alpha=0.5):
    """Superimposes the heatmap onto the original image."""
    try:
        img = cv2.imread(img_path)
        if img is None: raise ValueError("Could not read original image for display.")

        heatmap = np.uint8(255 * heatmap) # Scale to 0-255
        jet = plt.colormaps.get_cmap("jet") # Get the JET colormap
        jet_colors = jet(np.arange(256))[:, :3] # Get RGB values
        jet_heatmap = jet_colors[heatmap]

        # Convert to BGR for OpenCV
        jet_heatmap_bgr = cv2.cvtColor(np.float32(jet_heatmap), cv2.COLOR_RGB2BGR)

        # Resize heatmap to match original image
        jet_heatmap_resized = cv2.resize(jet_heatmap_bgr, (img.shape[1], img.shape[0]))

        superimposed_img = jet_heatmap_resized * alpha + img * (1-alpha)
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

        # Display
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
        plt.title(f'Grad-CAM for: {class_name}')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    except Exception as e:
        logging.error(f"Error displaying Grad-CAM: {e}", exc_info=True)

# =============================================================================
#                           --- Main Analysis ---
# =============================================================================
def main():
    logging.info("\n--- Analyzing Prediction with Grad-CAM ---")

    # --- Load Model FIRST ---
    logging.info(f"--- Loading Keras Model from {MODEL_PATH} ---")
    model = None # Initialize model as None
    if not os.path.exists(MODEL_PATH):
        logging.error(f"FATAL: Model file not found: {MODEL_PATH}")
        return # Exit if file not found
    else:
        try:
            model = keras.models.load_model(MODEL_PATH)
            logging.info(f"Keras model '{MODEL_FILENAME}' loaded successfully.")
        except Exception as e:
            logging.error(f"FATAL: Error loading Keras model: {e}", exc_info=True)
            return # Exit if loading fails
    # -------------------------

    # --- Load Char Map SECOND (or keep its loading block before model load) ---
    # (Char map loading logic needs to be here or above model loading)
    global char_map, text_vocab_size # Ensure globals are accessible if needed later
    char_map = {}; text_vocab_size = 1
    logging.info("--- Reconstructing Character Map ---")
    if os.path.exists(TRAIN_CSV_OCR):
        try:
            df_train = pd.read_csv(TRAIN_CSV_OCR); df_train['extracted_text'] = df_train['extracted_text'].fillna('')
            train_texts = df_train['extracted_text'].astype(str).tolist(); all_train_text = "".join(train_texts)
            unique_chars = sorted(list(set(all_train_text))); char_map = {char: i+1 for i, char in enumerate(unique_chars)}
            text_vocab_size = len(char_map) + 1; logging.info(f"Character map OK. Vocab size: {text_vocab_size}")
        except Exception as e: logging.error(f"Error char_map: {e}"); return # Exit on error
    else: logging.error(f"Training CSV ({TRAIN_CSV_OCR}) not found."); return # Exit on error
    # -----------------------------------------------------------------------


    # *** NOW perform the check ***
    if model is None or not char_map:
        # This condition should ideally not be met now if loading succeeded/exited above
        logging.error("Model or Character Map failed to load properly. Cannot analyze.")
        return
    # *****************************

    if not os.path.exists(IMAGE_TO_ANALYZE):
        logging.error(f"Image to analyze not found: {IMAGE_TO_ANALYZE}")
        return

    # --- Prepare Inputs ---
    logging.info("Preparing inputs...")
    # ... (rest of the input preparation: image for model, image for OCR, run OCR, text for model) ...
    preprocessed_image_model = preprocess_image_for_model(IMAGE_TO_ANALYZE)
    img_pil_for_ocr = preprocess_image_for_ocr_enhanced(IMAGE_TO_ANALYZE)
    ocr_text = ""
    if img_pil_for_ocr:
        try:
            ocr_text = pytesseract.image_to_string(img_pil_for_ocr, lang='eng', config='--psm 6')
            ocr_text = ocr_text.replace('\n', ' ').replace('\f', '').strip()
            logging.info(f"Tesseract OCR: '{ocr_text}'")
        except Exception as e: logging.warning(f"Tesseract failed: {e}")
    preprocessed_text_model = preprocess_text_for_model(ocr_text, char_map)


    if preprocessed_image_model is None or preprocessed_text_model is None:
        logging.error("Failed to preprocess inputs for analysis.")
        return

    # --- Get Prediction ---
    logging.info("Getting model prediction...")
    # ... (rest of prediction logic) ...
    prediction = model.predict([preprocessed_image_model, preprocessed_text_model], verbose=0)
    prediction_vector = prediction[0]
    predicted_class_index = np.argmax(prediction_vector)
    predicted_class_name = CLASS_NAMES[predicted_class_index]
    logging.info(f"Prediction: {predicted_class_name} (Probs: {prediction_vector})")


    # --- Generate and Display Grad-CAM ---
    print("\nGenerating Grad-CAM heatmap...")
    # ... (rest of Grad-CAM logic) ...
    target_class_index = None # Explain the predicted class
    try:
        # Find last conv layer name programmatically (safer)
        last_conv_layer_name = None
        for layer in reversed(model.layers):
             # Check for Conv2D or DepthwiseConv2D in the base model part
             if layer.name.startswith('block_') and isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D, tf.keras.layers.BatchNormalization, tf.keras.layers.Activation, tf.keras.layers.ReLU)):
                 # A heuristic: find the last relevant layer *before* GlobalAveragePooling
                 # Often a BN or activation layer associated with the last conv block's output
                 if layer.name == 'out_relu' or layer.name.endswith('_BN') or layer.name.endswith('_relu'):
                      # Go back slightly if needed to get the actual conv output if BN/ReLU is last
                      # This might need adjustment based on specific model structure (MobileNetV2's last block)
                      # Often 'block_16_project_BN' or 'out_relu' works for MobileNetV2
                      test_layer_name = 'out_relu' # Default guess for MobileNetV2
                      if model.get_layer(test_layer_name):
                          last_conv_layer_name = test_layer_name
                          break
                      # Fallback or specific name if needed
                      test_layer_name = 'block_16_project_BN'
                      if model.get_layer(test_layer_name):
                           last_conv_layer_name = test_layer_name
                           break

        if not last_conv_layer_name:
             logging.warning("Could not auto-detect last conv layer confidently. Using default 'out_relu'. Check model.summary().")
             last_conv_layer_name = 'out_relu' # Default fallback


        heatmap, explained_index = make_gradcam_heatmap(
            preprocessed_image_model,
            model,
            last_conv_layer_name,
            target_class_index,
            preprocessed_text_model
        )

        if heatmap is not None:
            print(f"Displaying Grad-CAM for predicted class: {CLASS_NAMES[explained_index]}")
            display_gradcam(IMAGE_TO_ANALYZE, heatmap, CLASS_NAMES[explained_index])
        else:
            print("Could not generate Grad-CAM heatmap.")

    except Exception as e:
            print(f"Error during Grad-CAM generation or display: {e}")
            print("--- Traceback ---"); traceback.print_exc(); print("-----------------")

    print("\n--- Analysis Complete ---")

# (keep the Grad-CAM functions and __main__ block)
if __name__ == "__main__":
    main()