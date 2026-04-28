# -*- coding: utf-8 -*-
"""
train_medicine_model_augmented_cpu.py

Builds, trains (with class weighting & text augmentation on CPU),
and evaluates a multi-modal model. Aims to make the model more robust
to potential OCR errors during inference.

Requirements:
  - tensorflow
  - opencv-python-headless
  - easyocr # For initial data prep only if needed
  - pandas
  - numpy
  - tqdm
  - matplotlib
  - scikit-learn
  - seaborn

Installation (in a virtual environment):
  pip install tensorflow opencv-python-headless easyocr pandas numpy tqdm matplotlib scikit-learn seaborn

Usage:
  1. Ensure CLEANED data is in place with _classes_with_ocr.csv files.
  2. Update BASE_PATH.
  3. Run from terminal: python train_medicine_model_augmented_cpu.py
"""
# --- Force CPU Usage for TensorFlow ---
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# -------------------------------------

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
# ... (rest of imports: cv2, numpy, pandas, math, tqdm, time, plt, zipfile, sklearn, seaborn) ...
import cv2
import numpy as np
import pandas as pd
import math
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import zipfile
import random # Needed for augmentation
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


# =============================================================================
#                           --- Configuration ---
# =============================================================================
# (Keep Config section the same, update BASE_PATH etc. as needed)
BASE_PATH = r"Caro_Laptop_Files"
IMG_WIDTH=224; IMG_HEIGHT=224; IMG_CHANNELS=3; BATCH_SIZE=32; EPOCHS=30
NUM_CLASSES=2; MAX_TEXT_LENGTH=50; TEXT_EMBEDDING_DIM=16; LEARNING_RATE=1e-4
IMAGENET_MEAN=np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD=np.array([0.229, 0.224, 0.225], dtype=np.float32)
CLASS_NAMES = ['Counterfeit', 'Authentic']

# --- Text Augmentation Config ---
TEXT_AUGMENT_PROB = 0.30 # Apply augmentation to ~30% of training text samples
CHAR_SWAP_PROB = 0.10   # Within an augmented sample, probability to swap a char
CHAR_DEL_PROB = 0.05    # Probability to delete a char
CHAR_INS_PROB = 0.05    # Probability to insert a char
# Define characters that might be confused by OCR
OCR_CONFUSABLES = {
    'O': '0', '0': 'O', 'I': '1', '1': 'I', 'L': '1', '1': 'L',
    'S': '5', '5': 'S', 'B': '8', '8': 'B', 'G': '6', '6': 'G',
    'Z': '2', '2': 'Z',
    # Add more as needed based on observed OCR errors
}
# =============================================================================


# --- Paths ---
# (Keep Paths section the same)
TRAIN_DIR = os.path.join(BASE_PATH, "train"); VALID_DIR = os.path.join(BASE_PATH, "valid"); TEST_DIR = os.path.join(BASE_PATH, "test")
TRAIN_CSV_OCR = os.path.join(TRAIN_DIR, "_classes_with_ocr.csv"); VALID_CSV_OCR = os.path.join(VALID_DIR, "_classes_with_ocr.csv"); TEST_CSV_OCR = os.path.join(TEST_DIR, "_classes_with_ocr.csv")
MODEL_CHECKPOINT_FILENAME = "mm_cmds_model_augmented_cpu_final.h5" # New name
MODEL_CHECKPOINT_PATH = os.path.join(BASE_PATH, MODEL_CHECKPOINT_FILENAME)
PLOT_FILENAME = "training_history_augmented_cpu.png" # New name
PLOT_PATH = os.path.join(BASE_PATH, PLOT_FILENAME)
# =============================================================================


# =============================================================================
#                           --- Helper Functions ---
# =============================================================================
# (Keep check_gpu_availability, load_data_from_processed_csv - unchanged)
# --- Re-paste necessary helper functions here for completeness ---
def check_gpu_availability():
    print("\n--- Checking Hardware ---"); print(f"TensorFlow Version: {tf.__version__}")
    if 'CUDA_VISIBLE_DEVICES' in os.environ and os.environ['CUDA_VISIBLE_DEVICES'] == '-1':
         print("GPU usage is explicitly disabled via CUDA_VISIBLE_DEVICES.")
         print(f"Visible Physical Devices: {tf.config.list_physical_devices()}")
         return False
    gpu_devices = tf.config.list_physical_devices('GPU')
    if gpu_devices:
        print(f"GPU Devices Detected: {gpu_devices}")
        try:
            for gpu in gpu_devices: tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU Memory Growth Enabled"); return True
        except RuntimeError as e: print(f"Warning: Could not set memory growth: {e}"); return True
    else: print("No GPU detected by TensorFlow. Using CPU."); return False

def load_data_from_processed_csv(csv_path, image_dir):
    # (Same as before)
    image_paths = []; texts = []; labels = []
    if not os.path.exists(csv_path): return [], [], []
    try:
        df = pd.read_csv(csv_path)
        if not all(col in df.columns for col in ['filename', 'label', 'extracted_text']): return [], [], []
        print(f"Loading data from {csv_path}...")
        df['extracted_text'] = df['extracted_text'].fillna('')
        num_skipped = 0
        for idx, row in df.iterrows():
            img_filename, label_val, text = row['filename'], int(row['label']), str(row['extracted_text'])
            img_path_base = os.path.join(image_dir, img_filename); actual_img_path = None
            if not os.path.splitext(img_path_base)[1]:
                for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                    potential_path = img_path_base + ext
                    if os.path.exists(potential_path): actual_img_path = potential_path; break
            elif os.path.exists(img_path_base): actual_img_path = img_path_base
            if actual_img_path: image_paths.append(actual_img_path); texts.append(text); labels.append(1 if label_val == 1 else 0)
            else: num_skipped += 1
        print(f"Loaded {len(image_paths)} samples. Skipped {num_skipped}.")
        return image_paths, texts, labels
    except Exception as e: print(f"Error loading {csv_path}: {e}"); return [], [], []

char_map_global = {}
text_vocab_size_global = 1

# --- Text Augmentation Functions ---
def random_char_swap(text_list):
    """Randomly swaps characters with visually similar ones."""
    if not text_list: return text_list
    idx = random.randint(0, len(text_list) - 1)
    char_to_swap = text_list[idx]
    if char_to_swap in OCR_CONFUSABLES and random.random() < CHAR_SWAP_PROB:
        text_list[idx] = OCR_CONFUSABLES[char_to_swap]
    return text_list

def random_char_delete(text_list):
    """Randomly deletes a character."""
    if not text_list: return text_list
    if random.random() < CHAR_DEL_PROB:
        idx = random.randint(0, len(text_list) - 1)
        del text_list[idx]
    return text_list

def random_char_insert(text_list):
    """Randomly inserts a character."""
    if random.random() < CHAR_INS_PROB and len(text_list) < MAX_TEXT_LENGTH: # Avoid exceeding max length
        idx = random.randint(0, len(text_list)) # Can insert at the end too
        # Insert a random char from the known vocab (or a space/common symbol)
        random_char = random.choice(list(char_map_global.keys()) + [' ', '.', '-'])
        text_list.insert(idx, random_char)
    return text_list

def augment_text_sequence(text):
    """Applies random augmentations to a text string."""
    text_list = list(text) # Work with list of characters
    # Apply augmentations sequentially (can be done in random order too)
    text_list = random_char_swap(text_list)
    text_list = random_char_delete(text_list)
    text_list = random_char_insert(text_list)
    # Add more augmentation types here if needed
    return "".join(text_list)

# --- Modified TF Data Pipeline Functions ---

# Image preprocessing function remains the same
def _load_and_preprocess_image_py(image_path_tensor):
    # (Same as before)
    image_path = image_path_tensor.numpy().decode('utf-8')
    try:
        img = cv2.imread(image_path)
        if img is None: return np.zeros((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB); img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img = img.astype(np.float32) / 255.0; img = (img - IMAGENET_MEAN) / IMAGENET_STD
        return img
    except Exception: return np.zeros((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)

# Text encoding function now incorporates OPTIONAL augmentation
def _encode_and_maybe_augment_text_py(text_tensor, augment_prob_tensor):
    text = text_tensor.numpy().decode('utf-8')
    augment_prob = augment_prob_tensor.numpy() # Get probability value

    # Apply augmentation based on probability
    if random.random() < augment_prob:
        text = augment_text_sequence(text) # Augment the raw text string

    # Preprocess and Encode the (potentially augmented) text
    processed_text = text.strip().upper()[:MAX_TEXT_LENGTH]
    encoded = np.zeros((MAX_TEXT_LENGTH,), dtype=np.int32)
    for i, char in enumerate(processed_text):
        encoded[i] = char_map_global.get(char, 0) # Use global char_map

    # Padding is handled separately by Keras layers or explicitly if needed before model
    # For direct input like this, explicit padding here is safer
    # Re-using pad_sequences (ensure numpy input)
    padded_encoded = pad_sequences([encoded], maxlen=MAX_TEXT_LENGTH, padding='post', truncating='post')[0]

    return padded_encoded # Return the numpy array


@tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.string)])
def tf_load_preprocess_image(image_path_tensor):
    # (Same as before)
    image = tf.py_function(func=_load_and_preprocess_image_py, inp=[image_path_tensor], Tout=tf.float32)
    image.set_shape((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)); return image

# Modified TF text wrapper to accept probability
@tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.string), tf.TensorSpec(shape=(), dtype=tf.float32)])
def tf_encode_text(text_tensor, augment_prob_tensor):
    encoded_text = tf.py_function(
        func=_encode_and_maybe_augment_text_py,
        inp=[text_tensor, augment_prob_tensor], # Pass probability in
        Tout=tf.int32
    )
    encoded_text.set_shape((MAX_TEXT_LENGTH,)); return encoded_text

# Modified dataset creation to handle augmentation flag
def create_tf_dataset(image_paths, texts, labels, is_training=False):
    """Creates tf.data.Dataset, optionally applying text augmentation."""
    labels = tf.cast(labels, dtype=tf.int32)
    dataset = tf.data.Dataset.from_tensor_slices(((image_paths, texts), labels))

    def load_and_preprocess(inputs, label):
        image_path, text = inputs
        processed_image = tf_load_preprocess_image(image_path)

        # Determine augmentation probability based on is_training flag
        augment_prob = tf.constant(TEXT_AUGMENT_PROB, dtype=tf.float32) if is_training else tf.constant(0.0, dtype=tf.float32)
        encoded_text = tf_encode_text(text, augment_prob) # Pass probability tensor

        one_hot_label = tf.one_hot(label, depth=NUM_CLASSES)
        return (processed_image, encoded_text), one_hot_label

    dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset

# (Keep build_mm_cmds_model function - same as before)
def build_mm_cmds_model(num_classes, max_text_len, vocab_size, embedding_dim):
    # (Same model: MobileNetV2 + GRU(reset_after=False))
    image_input = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), name="image_input")
    base_model = tf.keras.applications.MobileNetV2(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), include_top=False, weights='imagenet', input_tensor=image_input)
    base_model.trainable = False
    x = base_model.output; x = layers.GlobalAveragePooling2D(name="image_pooling")(x)
    x = layers.BatchNormalization()(x); x = layers.Dropout(0.3, name="image_dropout")(x)
    image_features = layers.Dense(128, activation='relu', name="image_features")(x)
    text_input = layers.Input(shape=(max_text_len,), name="text_input", dtype=tf.int32)
    t = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True, name="text_embedding")(text_input)
    t = layers.GRU(64, return_sequences=False, name="text_gru", reset_after=False)(t)
    t = layers.BatchNormalization()(t); t = layers.Dropout(0.3, name="text_dropout")(t)
    text_features = layers.Dense(64, activation='relu', name="text_features")(t)
    fused = layers.Concatenate(name="fusion")([image_features, text_features])
    fused = layers.BatchNormalization()(fused); fused = layers.Dense(64, activation='relu', name="fusion_dense")(fused)
    fused = layers.Dropout(0.5, name="fusion_dropout")(fused)
    output = layers.Dense(num_classes, activation='softmax', name="output")(fused)
    model = keras.Model(inputs=[image_input, text_input], outputs=output, name="MM_CMDS_GRU_StdKernel_Aug") # New name
    return model

# (Keep plot_training_history function - same as before)
def plot_training_history(history, save_path):
    # ... (plotting code remains the same) ...
    acc=history.history['accuracy']; val_acc=history.history['val_accuracy']
    loss=history.history['loss']; val_loss=history.history['val_loss']
    epochs_range = range(len(acc))
    plt.figure(figsize=(12, 5)); plt.subplot(1, 2, 1); plt.plot(epochs_range, acc, label='Training Accuracy'); plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right'); plt.title('Training and Validation Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.grid(True)
    plt.subplot(1, 2, 2); plt.plot(epochs_range, loss, label='Training Loss'); plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right'); plt.title('Training and Validation Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.grid(True)
    plt.tight_layout();
    try: plt.savefig(save_path); print(f"\nTraining history plot saved to: {save_path}")
    except Exception as e: print(f"\nError saving training plot: {e}")
# =============================================================================


# =============================================================================
#                           --- Main Execution ---
# =============================================================================

def main():
    """Main function for training with text augmentation and class weighting."""
    global char_map_global, text_vocab_size_global # Allow modification

    print("--- Running Training (Class Weighting & Text Augmentation - FORCED CPU) ---")
    print(f"Base Path: {BASE_PATH}")
    if not os.path.exists(BASE_PATH): return

    print("\n" + "="*30); print("IMPORTANT: Ensure CLEANED train/valid data!"); print("="*30 + "\n"); time.sleep(1)
    has_gpu = check_gpu_availability() # Will report disabled

    # --- Phase 1: Load Data ---
    print("\n--- Phase 1: Loading Data ---")
    train_image_paths, train_texts, train_labels_raw = load_data_from_processed_csv(TRAIN_CSV_OCR, TRAIN_DIR)
    valid_image_paths, valid_texts, valid_labels_raw = load_data_from_processed_csv(VALID_CSV_OCR, VALID_DIR)
    test_image_paths, test_texts, test_labels_raw = [], [], []
    if os.path.exists(TEST_CSV_OCR): test_image_paths, test_texts, test_labels_raw = load_data_from_processed_csv(TEST_CSV_OCR, TEST_DIR)
    if not train_image_paths or not valid_image_paths: print("\nError loading data."); return

    # --- Phase 2: Data Preparation ---
    print("\n--- Phase 2: Data Preparation ---")
    print("Building character map..."); all_train_text = "".join(train_texts); unique_chars = sorted(list(set(all_train_text)))
    char_map_global = {char: i+1 for i, char in enumerate(unique_chars)}; text_vocab_size_global = len(char_map_global) + 1
    print(f"Character map built. Vocab size: {text_vocab_size_global}")

    print("Calculating class weights..."); neg_count = sum(1 for l in train_labels_raw if l == 0); pos_count = len(train_labels_raw) - neg_count
    total_count = len(train_labels_raw); calculated_class_weights = {0: 1.0, 1: 1.0}
    if neg_count > 0 and pos_count > 0:
        weight_for_0 = (total_count/(2.0*neg_count)); weight_for_1 = (total_count/(2.0*pos_count))
        calculated_class_weights = {0: weight_for_0, 1: weight_for_1}
        print(f"  Counts C(0):{neg_count}, A(1):{pos_count} -> Weights: {calculated_class_weights}")
    else: print("Warning: Zero samples for one class. Using equal weights.")

    print("\nBuilding tf.data Pipelines (with training augmentation)...");
    # *** Apply augmentation ONLY to training set ***
    train_dataset = create_tf_dataset(train_image_paths, train_texts, train_labels_raw, is_training=True)
    train_dataset = train_dataset.shuffle(buffer_size=max(1000, len(train_image_paths)//4)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    valid_dataset = create_tf_dataset(valid_image_paths, valid_texts, valid_labels_raw, is_training=False) # No augmentation
    valid_dataset = valid_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    test_dataset = None
    if test_image_paths:
        test_dataset = create_tf_dataset(test_image_paths, test_texts, test_labels_raw, is_training=False) # No augmentation
        test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        print("Train, Validation, and Test datasets created.")
    else: print("Train and Validation datasets created (No test data).")

    # --- Phase 3: Model Definition ---
    print("\n--- Phase 3: Model Definition ---")
    model = build_mm_cmds_model(NUM_CLASSES, MAX_TEXT_LENGTH, text_vocab_size_global, TEXT_EMBEDDING_DIM)
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='categorical_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision(name='precision'),
                  tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.AUC(name='auc')])
    print("Model Compiled.")

    # --- Phase 4: Model Training ---
    print("\n--- Phase 4: Model Training (Weights + Augmentation - CPU) ---")
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, min_lr=1e-7, verbose=1)
    print(f"Setting checkpoint path to: {MODEL_CHECKPOINT_PATH}")
    checkpoint = tf.keras.callbacks.ModelCheckpoint(MODEL_CHECKPOINT_PATH, monitor='val_auc', save_best_only=True, mode='max', verbose=1)
    print(f"Training samples: {len(train_image_paths)}"); print(f"Validation samples: {len(valid_image_paths)}")
    print(f"Batch size: {BATCH_SIZE}, Target Epochs: {EPOCHS}")

    print("\nStarting training...")
    start_train_time = time.time()
    history = model.fit(train_dataset, epochs=EPOCHS, validation_data=valid_dataset,
                        callbacks=[early_stopping, reduce_lr, checkpoint],
                        class_weight=calculated_class_weights, verbose=1)
    end_train_time = time.time()
    print(f"Training finished. Time taken: {end_train_time - start_train_time:.2f} seconds.")

    if history and history.history: plot_training_history(history, PLOT_PATH)

    # --- Phase 5: Evaluation ---
    # (Evaluation code remains largely the same, just loads the new model)
    print("\n--- Phase 5: Model Evaluation ---")
    print(f"Loading best model based on val_auc from: {MODEL_CHECKPOINT_PATH}")
    if os.path.exists(MODEL_CHECKPOINT_PATH):
        try:
            #if not zipfile.is_zipfile(MODEL_CHECKPOINT_PATH): raise ValueError("Saved model file is corrupted.")
            best_model = keras.models.load_model(MODEL_CHECKPOINT_PATH)
            print("Best model loaded successfully.")
            # ... (Rest of evaluation, including detailed report generation) ...
            print("\nEvaluating on Validation Set:")
            val_results = best_model.evaluate(valid_dataset, verbose=0, return_dict=True)
            print(f"  Validation Loss: {val_results['loss']:.4f}, Accuracy: {val_results['accuracy']:.4f}, Precision: {val_results['precision']:.4f}, Recall: {val_results['recall']:.4f}, AUC: {val_results['auc']:.4f}")

            if test_dataset:
                print("\nEvaluating on Test Set:")
                test_results = best_model.evaluate(test_dataset, verbose=0, return_dict=True)
                print(f"  Test Loss: {test_results['loss']:.4f}, Accuracy: {test_results['accuracy']:.4f}, Precision: {test_results['precision']:.4f}, Recall: {test_results['recall']:.4f}, AUC: {test_results['auc']:.4f}")
                print("\nGenerating Detailed Classification Report on Test Set...")
                try:
                    print("  Running predictions..."); y_pred_probs = best_model.predict(test_dataset)
                    y_pred_indices = np.argmax(y_pred_probs, axis=1)
                    print("  Extracting true labels..."); y_true_indices = []
                    for _, labels_batch in test_dataset: y_true_indices.extend(np.argmax(labels_batch.numpy(), axis=1)) # Corrected extraction

                    if len(y_true_indices) == len(y_pred_indices):
                         print("\nClassification Report (Test Set):")
                         report = classification_report(y_true_indices, y_pred_indices, target_names=CLASS_NAMES, digits=4) # Increased digits
                         print(report)
                         print("\nConfusion Matrix (Test Set):")
                         cm = confusion_matrix(y_true_indices, y_pred_indices); print(cm)
                         plt.figure(figsize=(6, 5)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
                         plt.xlabel('Predicted Label'); plt.ylabel('True Label'); plt.title('Confusion Matrix - Augmented CPU Model')
                         cm_save_path = os.path.join(BASE_PATH, "confusion_matrix_augmented_cpu.png") # New name
                         plt.savefig(cm_save_path); print(f"Confusion matrix plot saved to: {cm_save_path}")
                    else: print(f"Error: Label count mismatch ({len(y_true_indices)} vs {len(y_pred_indices)})")
                except ImportError: print("Warning: scikit-learn or seaborn not installed.")
                except Exception as report_err: print(f"Error during detailed report generation: {report_err}")
            else: print("\nNo test set available for final evaluation.")

        except Exception as e: print(f"Error loading or evaluating the best model: {e}")
    else: print(f"Model checkpoint file not found at {MODEL_CHECKPOINT_PATH}. Evaluation skipped.")

    print("\n--- Script Execution Complete ---")

if __name__ == "__main__":
    main()