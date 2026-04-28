# -*- coding: utf-8 -*-
"""
** current **
tune_threshold.py

Loads the previously trained weighted model and evaluates performance
on the validation set using different prediction thresholds for the
'Counterfeit' class to find a potentially better balance between
precision and recall.

Requirements:
  - tensorflow
  - numpy
  - pandas
  - scikit-learn
  - opencv-python-headless (indirectly via data loading funcs)
  - matplotlib
"""

# --- Force CPU Usage ---
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# -------------------------------------

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # Needed for model definition if loading weights only
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
import cv2
import math
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, precision_recall_curve, roc_curve, auc 
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

# =============================================================================
#                           --- Configuration ---
# =============================================================================
# --- Model & Data Constants ---
# (Must match the settings used for training the loaded model)
IMG_WIDTH=224; IMG_HEIGHT=224; IMG_CHANNELS=3; MAX_TEXT_LENGTH=50
NUM_CLASSES=2; TEXT_EMBEDDING_DIM=16;
IMAGENET_MEAN=np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD=np.array([0.229, 0.224, 0.225], dtype=np.float32)
CLASS_NAMES = ['Counterfeit', 'Authentic'] # 0, 1

# --- Paths ---
BASE_PATH = r"Caro_Laptop_Files_CROPPED"
# *** USE THE WEIGHTED MODEL (NOT the auth-aug-as-fake one) ***
MODEL_FILENAME = r"mm_cmds_model_paddle_ft_robust_v4.h5"  # Or "..._final.keras" if that was the good one
MODEL_PATH = os.path.join(BASE_PATH, MODEL_FILENAME)

# --- Paths for Data Loading & Char Map ---
TRAIN_DIR = os.path.join(BASE_PATH, "train") # Needed for char_map
TRAIN_CSV_OCR = os.path.join(TRAIN_DIR, "_classes_with_paddle_ocr.csv")
VALID_DIR = os.path.join(BASE_PATH, "valid") # Evaluate on validation set
VALID_CSV_OCR = os.path.join(VALID_DIR, "_classes_with_paddle_ocr.csv")

BATCH_SIZE = 32 # Use a reasonable batch size for prediction/evaluation
# =============================================================================


# =============================================================================
#            --- Char Map, Preprocessing & Dataset Functions ---
# =============================================================================
# (Need the same functions used during training to create the validation dataset)
char_map_global = {}
text_vocab_size_global = 1

def load_data_from_processed_csv(csv_path, image_dir):
    # (Same as in training script)
    image_paths, texts, labels = [], [], []
    if not os.path.exists(csv_path): return [], [], []
    try:
        df = pd.read_csv(csv_path)
        if not all(col in df.columns for col in ['filename', 'label', 'extracted_text']): return [], [], []
        logging.info(f"Loading data from {csv_path}...")
        df['extracted_text'] = df['extracted_text'].fillna('')
        num_skipped = 0
        for _, row in df.iterrows():
            img_filename, label_val, text = row['filename'], int(row['label']), str(row['extracted_text'])
            img_path_base = os.path.join(image_dir, img_filename); actual_img_path = None
            if not os.path.splitext(img_path_base)[1]:
                for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                    potential_path = img_path_base + ext
                    if os.path.exists(potential_path): actual_img_path = potential_path; break
            elif os.path.exists(img_path_base): actual_img_path = img_path_base
            if actual_img_path: image_paths.append(actual_img_path); texts.append(text); labels.append(1 if label_val == 1 else 0)
            else: num_skipped += 1
        logging.info(f"Loaded {len(image_paths)} samples. Skipped {num_skipped}.")
        return image_paths, texts, labels
    except Exception as e: logging.error(f"Error loading {csv_path}: {e}"); return [], [], []

def _load_and_preprocess_image_py(image_path_tensor):
    # (Same as in training script)
    image_path = image_path_tensor.numpy().decode('utf-8')
    try:
        img = cv2.imread(image_path); img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB); img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img = img.astype(np.float32) / 255.0; img = (img - IMAGENET_MEAN) / IMAGENET_STD
        return img
    except Exception: return np.zeros((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)

def _encode_text_py(text_tensor): # Simple version without augmentation
    text = text_tensor.numpy().decode('utf-8')
    processed_text = text.strip().upper()[:MAX_TEXT_LENGTH]
    encoded = np.zeros((MAX_TEXT_LENGTH,), dtype=np.int32)
    for i, char in enumerate(processed_text): encoded[i] = char_map_global.get(char, 0)
    padded_encoded = pad_sequences([encoded], maxlen=MAX_TEXT_LENGTH, padding='post', truncating='post')[0]
    return padded_encoded

@tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.string)])
def tf_load_preprocess_image(image_path_tensor):
    # (Same as in training script)
    image = tf.py_function(func=_load_and_preprocess_image_py, inp=[image_path_tensor], Tout=tf.float32)
    image.set_shape((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)); return image

@tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.string)])
def tf_encode_text_simple(text_tensor): # Renamed for clarity
    encoded_text = tf.py_function(func=_encode_text_py, inp=[text_tensor], Tout=tf.int32)
    encoded_text.set_shape((MAX_TEXT_LENGTH,)); return encoded_text

def create_tf_dataset_for_eval(image_paths, texts, labels):
    """Creates dataset for evaluation (no augmentation/relabeling)."""
    labels_tensor = tf.cast(labels, dtype=tf.int32)
    dataset = tf.data.Dataset.from_tensor_slices(((image_paths, texts), labels_tensor))
    def load_and_preprocess(inputs, label):
        image_path, text = inputs
        processed_image = tf_load_preprocess_image(image_path)
        encoded_text = tf_encode_text_simple(text) # Use simple encoding
        # Return the original label (0 or 1) along with processed inputs
        return (processed_image, encoded_text), label
    dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset
# =============================================================================


# =============================================================================
#                           --- Main Logic ---
# =============================================================================
def main():
    global char_map_global, text_vocab_size_global # Allow modification

    print("--- Threshold Tuning for Weighted Model ---")
    # --- Load Char Map ---
    logging.info("--- Reconstructing Character Map ---")
    if os.path.exists(TRAIN_CSV_OCR):
        try:
            df_train = pd.read_csv(TRAIN_CSV_OCR); df_train['extracted_text'] = df_train['extracted_text'].fillna('')
            train_texts = df_train['extracted_text'].astype(str).tolist(); all_train_text = "".join(train_texts)
            unique_chars = sorted(list(set(all_train_text))); char_map_global = {char: i+1 for i, char in enumerate(unique_chars)}
            text_vocab_size_global = len(char_map_global) + 1; logging.info(f"Char map OK. Vocab: {text_vocab_size_global}")
        except Exception as e: logging.error(f"Error reconstructing char_map: {e}"); return
    else: logging.error(f"Training CSV ({TRAIN_CSV_OCR}) not found."); return

    # --- Load Model ---
    logging.info(f"--- Loading Keras Model from {MODEL_PATH} ---")
    if not os.path.exists(MODEL_PATH): logging.error(f"Model file not found."); return
    try:
        model = keras.models.load_model(MODEL_PATH)
        logging.info("Model loaded successfully.")
    except Exception as e: logging.error(f"Error loading model: {e}"); return

    # --- Load Validation Data ---
    print("\n--- Loading Validation Data ---")
    valid_image_paths, valid_texts, valid_labels_raw = load_data_from_processed_csv(VALID_CSV_OCR, VALID_DIR)
    if not valid_image_paths: print("No validation data loaded."); return

    print("\nBuilding Validation Dataset Pipeline...")
    # Create dataset that yields ((img, txt), label_0_or_1)
    val_dataset = create_tf_dataset_for_eval(valid_image_paths, valid_texts, valid_labels_raw)
    val_dataset_batched = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # --- Get Predictions and True Labels ---
    print("\nRunning predictions on validation set...")
    y_pred_probs_list = []
    y_true_list = []
    for batch_data, batch_labels in tqdm(val_dataset_batched, desc="Predicting"):
        preds = model.predict(batch_data, verbose=0)
        y_pred_probs_list.extend(preds)
        y_true_list.extend(batch_labels.numpy()) # Store original 0/1 labels

    y_pred_probs = np.array(y_pred_probs_list)
    y_true = np.array(y_true_list)

    # Extract probability for the POSITIVE class (Authentic = 1)
    # If we threshold the COUNTERFEIT class (index 0), we need those probabilities
    y_scores_counterfeit = y_pred_probs[:, 0] # Probabilities for class 0

    # --- Iterate Through Thresholds ---
    print("\n--- Evaluating Thresholds for Counterfeit Class (0) ---")
    thresholds = np.arange(0.1, 1.0, 0.05) # Check thresholds from 0.1 to 0.95
    results = []

    for thresh in thresholds:
        # Predict class 0 if probability > threshold
        y_pred_thresh = (y_scores_counterfeit >= thresh).astype(int)
        # Note: This means 1 = Predicted Counterfeit, 0 = Predicted Authentic under this logic

        # Calculate metrics for the *original* labels (0=C, 1=A) vs thresholded predictions
        # We want precision/recall for class 0 (Counterfeit)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,        # True labels (0=C, 1=A)
            1 - y_pred_thresh, # Predicted labels (invert: 0=C, 1=A)
            labels=[0, 1], # Evaluate both classes
            zero_division=0
        )
        # Get metrics specifically for Counterfeit (label 0)
        precision_c = precision[0]
        recall_c = recall[0]
        f1_c = f1[0]

        # Calculate overall accuracy for this threshold
        accuracy = np.mean( (1 - y_pred_thresh) == y_true)

        cm = confusion_matrix(y_true, 1 - y_pred_thresh)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0) # Handle potential edge cases

        results.append({
            'Threshold': thresh,
            'Accuracy': accuracy,
            'C_Precision': precision_c,
            'C_Recall': recall_c,
            'C_F1': f1_c,
            'False_Positives (Auth as C)': fp, # FP for class 1 (Authentic) are TN for class 0
            'False_Negatives (C as Auth)': fn  # FN for class 1 are TP for class 0
             # Confusion matrix values based on standard layout (rows=true, cols=pred)
             # for labels 0, 1: TN=cm[0,0], FP=cm[0,1], FN=cm[1,0], TP=cm[1,1]
             # Let's be explicit for Counterfeit (class 0):
             # TP_C = cm[0,0] # True Counterfeit correctly predicted
             # FP_C = cm[1,0] # False Counterfeit (Authentic predicted as C)
             # FN_C = cm[0,1] # False Authentic (Counterfeit predicted as A)
             # TN_C = cm[1,1] # True Authentic correctly predicted
             # Let's use the standard sklearn output names for clarity
        })
        print(f"Threshold >= {thresh:.2f}: Acc={accuracy:.4f}, C_Prec={precision_c:.4f}, C_Recall={recall_c:.4f}, C_F1={f1_c:.4f}, FP(Auth as C)={fp}, FN(C as Auth)={fn}")


    # --- Analyze Results ---
    results_df = pd.DataFrame(results)
    print("\n--- Results Summary ---")
    print(results_df.to_string())

    # Find threshold maximizing F1 for Counterfeit, while keeping Recall high?
    # Example: Find best F1 where Counterfeit Recall is >= 0.9
    best_f1_high_recall = results_df[results_df['C_Recall'] >= 0.90].sort_values('C_F1', ascending=False)

    print("\n--- Suggested Thresholds (Based on Validation Set) ---")
    if not best_f1_high_recall.empty:
        best_threshold_f1 = best_f1_high_recall.iloc[0]
        print(f"Best F1 for Counterfeit (Recall >= 0.90):")
        print(f"  Threshold: {best_threshold_f1['Threshold']:.2f}")
        print(f"  Precision: {best_threshold_f1['C_Precision']:.4f}")
        print(f"  Recall:    {best_threshold_f1['C_Recall']:.4f}")
        print(f"  F1-Score:  {best_threshold_f1['C_F1']:.4f}")
        print(f"  Accuracy:  {best_threshold_f1['Accuracy']:.4f}")
    else:
        print("Could not find threshold with Counterfeit Recall >= 0.90. Check results.")

    # Find threshold maximizing precision, while keeping Recall >= 0.8?
    best_prec_good_recall = results_df[results_df['C_Recall'] >= 0.80].sort_values('C_Precision', ascending=False)
    if not best_prec_good_recall.empty:
        best_threshold_prec = best_prec_good_recall.iloc[0]
        print(f"\nBest Precision for Counterfeit (Recall >= 0.80):")
        print(f"  Threshold: {best_threshold_prec['Threshold']:.2f}")
        print(f"  Precision: {best_threshold_prec['C_Precision']:.4f}")
        print(f"  Recall:    {best_threshold_prec['C_Recall']:.4f}")
        print(f"  F1-Score:  {best_threshold_prec['C_F1']:.4f}")
        print(f"  Accuracy:  {best_threshold_prec['Accuracy']:.4f}")
    else:
        print("\nCould not find threshold with Counterfeit Recall >= 0.80. Check results.")

    # --- Plot Precision-Recall Curve ---
    precision_curve, recall_curve, thresholds_pr = precision_recall_curve(y_true, y_scores_counterfeit, pos_label=0) # Curve for Counterfeit class

    plt.figure(figsize=(8, 6))
    plt.plot(recall_curve, precision_curve, marker='.', label='Precision-Recall Curve (Counterfeit)')
    # Add point for default 0.5 threshold performance
    # Need to calculate precision/recall at 0.5 explicitly
    y_pred_05 = (y_scores_counterfeit >= 0.5).astype(int)
    precision_05, recall_05, _, _ = precision_recall_fscore_support(y_true, 1-y_pred_05, labels=[0], zero_division=0, average='binary')
    plt.scatter([recall_05], [precision_05], marker='o', color='red', label=f'Default Threshold (0.5)\nPrec={precision_05:.2f}, Rec={recall_05:.2f}')
    plt.xlabel('Recall (Counterfeit)')
    plt.ylabel('Precision (Counterfeit)')
    plt.title('Counterfeit Class Precision-Recall Curve (Validation Set)')
    plt.legend()
    plt.grid(True)
    pr_curve_path = os.path.join(BASE_PATH, "precision_recall_curve.png")
    plt.savefig(pr_curve_path)
    print(f"\nPrecision-Recall curve saved to: {pr_curve_path}")
    # plt.show()

    print("\n--- Threshold Tuning Complete ---")
    print("Recommendation: Choose a threshold based on the table above and the P-R curve, considering the balance your application needs.")

    # --- Plot ROC Curve ---
    print("\nGenerating ROC Curve...")
    # Calculate ROC curve points
    # Use y_true (original labels 0=C, 1=A)
    # Use y_scores_counterfeit (probability of class 0 - Counterfeit)
    # Note: roc_curve expects scores for the POSITIVE class.
    # If we want the curve for detecting COUNTERFEIT (class 0) as positive:
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_scores_counterfeit, pos_label=0)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance (AUC = 0.5)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR / Recall)')
    plt.title('Receiver Operating Characteristic (ROC) Curve - Counterfeit Class (Validation)')
    plt.legend(loc="lower right")
    plt.grid(True)
    roc_curve_path = os.path.join(BASE_PATH, "roc_curve.png")
    try:
        plt.savefig(roc_curve_path)
        print(f"ROC curve saved to: {roc_curve_path}")
    except Exception as e:
        print(f"Error saving ROC curve plot: {e}")
    # plt.show()

if __name__ == "__main__":
    main()