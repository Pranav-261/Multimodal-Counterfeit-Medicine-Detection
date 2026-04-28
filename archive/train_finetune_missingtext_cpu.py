# -*- coding: utf-8 -*-
"""
train_finetune_missingtext_cpu.py

Trains the model by:
1. Fine-tuning later layers of the visual base model (MobileNetV2).
2. Applying class weighting.
3. Training on augmented text, including replacing text with empty/garbage
   sequences to simulate OCR failure, while keeping original labels.
4. Runs entirely on CPU.

Requirements: [Same as before]
"""
# --- Force CPU Usage ---
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# -------------------------------------

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
import cv2
import numpy as np
import pandas as pd
import math
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import zipfile
import random
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import traceback
import string # For generating garbage text

# =============================================================================
#                           --- Configuration ---
# =============================================================================
BASE_PATH = r"Caro_Laptop_Files"
IMG_WIDTH=224; IMG_HEIGHT=224; IMG_CHANNELS=3; BATCH_SIZE=32; EPOCHS=25 # Reduced epochs slightly due to slower fine-tuning
NUM_CLASSES=2; MAX_TEXT_LENGTH=50; TEXT_EMBEDDING_DIM=16;

# *** Fine-tuning LEARNING RATE - MUST be low ***
LEARNING_RATE = 1e-5 # Start with a low learning rate for fine-tuning

IMAGENET_MEAN=np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD=np.array([0.229, 0.224, 0.225], dtype=np.float32)
CLASS_NAMES = ['Counterfeit', 'Authentic']

# --- Text Augmentation Config ---
# Probabilities to apply ANY text modification (Empty, Garbage, or Char-level Aug)
APPLY_TEXT_MOD_PROB = 0.30 # Modify text for ~30% of training samples
# Within those modified samples, the breakdown:
MISSING_TEXT_PROB = 0.40   # Of the modified, 40% become empty string
GARBAGE_TEXT_PROB = 0.30   # Of the modified, 30% become random garbage
# Implicitly, 100-40-30 = 30% undergo char-level augmentation

# Probabilities for character-level augmentation (if applied)
CHAR_SWAP_PROB = 0.15
CHAR_DEL_PROB = 0.10
CHAR_INS_PROB = 0.10
OCR_CONFUSABLES = {
    'O': '0', '0': 'O', 'I': '1', '1': 'I', 'L': '1', '1': 'L', 'S': '5', '5': 'S',
    'B': '8', '8': 'B', 'G': '6', '6': 'G', 'Z': '2', '2': 'Z',
}
GARBAGE_CHARS = string.ascii_uppercase + string.digits + string.punctuation + ' '

# --- Fine-tuning Config ---
# Unfreeze layers starting from this block in MobileNetV2
# Find layer index by running base_model.summary() and looking for block names
# e.g., 'block_13_expand' is often a reasonable point
FINE_TUNE_AT_BLOCK = 'block_13_expand'
# =============================================================================


# --- Paths ---
TRAIN_DIR = os.path.join(BASE_PATH, "train"); VALID_DIR = os.path.join(BASE_PATH, "valid"); TEST_DIR = os.path.join(BASE_PATH, "test")
TRAIN_CSV_OCR = os.path.join(TRAIN_DIR, "_classes_with_ocr.csv"); VALID_CSV_OCR = os.path.join(VALID_DIR, "_classes_with_ocr.csv"); TEST_CSV_OCR = os.path.join(TEST_DIR, "_classes_with_ocr.csv")
# *** NEW FILENAMES ***
MODEL_CHECKPOINT_FILENAME = "mm_cmds_model_finetune_miss_cpu.h5" # FineTune+MissingText
MODEL_CHECKPOINT_PATH = os.path.join(BASE_PATH, MODEL_CHECKPOINT_FILENAME)
PLOT_FILENAME = "training_history_finetune_miss_cpu.png"
PLOT_PATH = os.path.join(BASE_PATH, PLOT_FILENAME)
CM_FILENAME = "confusion_matrix_finetune_miss_cpu.png"
CM_SAVE_PATH = os.path.join(BASE_PATH, CM_FILENAME)
# =============================================================================

# =============================================================================
#                           --- Helper Functions ---
# =============================================================================
# (Keep check_gpu_availability, load_data_from_processed_csv - unchanged)
# (Keep plot_training_history - unchanged)
# --- Re-paste necessary helper functions here ---
def check_gpu_availability():
    print("\n--- Checking Hardware ---"); print(f"TensorFlow Version: {tf.__version__}")
    if 'CUDA_VISIBLE_DEVICES' in os.environ and os.environ['CUDA_VISIBLE_DEVICES'] == '-1':
         print("GPU usage is explicitly disabled via CUDA_VISIBLE_DEVICES.")
         print(f"Visible Physical Devices: {tf.config.list_physical_devices()}")
         return False
    gpu_devices = tf.config.list_physical_devices('GPU')
    if gpu_devices: print(f"GPU Devices Detected: {gpu_devices}"); return True # Still return True if detected
    else: print("No GPU detected by TensorFlow. Using CPU."); return False

def load_data_from_processed_csv(csv_path, image_dir):
    # (Same as before - loads raw 0/1 labels)
    image_paths, texts, labels = [], [], []
    if not os.path.exists(csv_path): return [], [], []
    try:
        df = pd.read_csv(csv_path); df['extracted_text'] = df['extracted_text'].fillna('')
        print(f"Loading data from {csv_path}...")
        num_skipped = 0
        for _, row in df.iterrows():
            # ... (same file checking logic) ...
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

# --- Text Augmentation Functions (Including Empty/Garbage) ---
def random_char_swap(text_list):
    if not text_list: return text_list
    idx = random.randint(0, len(text_list)-1); char_to_swap = text_list[idx]
    if char_to_swap in OCR_CONFUSABLES and random.random() < CHAR_SWAP_PROB:
        text_list[idx] = OCR_CONFUSABLES[char_to_swap]
    return text_list
def random_char_delete(text_list):
    if not text_list: return text_list
    if random.random() < CHAR_DEL_PROB and len(text_list) > 1: # Avoid deleting last char
        idx = random.randint(0, len(text_list)-1); del text_list[idx]
    return text_list
def random_char_insert(text_list):
    if random.random() < CHAR_INS_PROB and len(text_list) < MAX_TEXT_LENGTH:
        idx = random.randint(0, len(text_list))
        random_char = random.choice(list(char_map_global.keys()) + [' ','.','-'])
        text_list.insert(idx, random_char)
    return text_list
def generate_garbage_text(length):
     return ''.join(random.choice(GARBAGE_CHARS) for _ in range(length))

def augment_text_missing_garbage(text):
    """Applies random Empty, Garbage, or Char-Level augmentations."""
    mod_type = random.random() # Decide modification type

    if mod_type < MISSING_TEXT_PROB:
        return "" # Replace with empty string
    elif mod_type < MISSING_TEXT_PROB + GARBAGE_TEXT_PROB:
        # Replace with garbage, random length up to MAX_TEXT_LENGTH
        garbage_len = random.randint(5, MAX_TEXT_LENGTH) # Make garbage reasonably long
        return generate_garbage_text(garbage_len)
    else:
        # Apply character-level augmentations
        text_list = list(text)
        text_list = random_char_swap(text_list)
        text_list = random_char_delete(text_list)
        text_list = random_char_insert(text_list)
        return "".join(text_list)

# --- TF Data Pipeline Functions ---
def _load_and_preprocess_image_py(image_path_tensor):
    # (Same as before)
    image_path = image_path_tensor.numpy().decode('utf-8')
    try:
        img = cv2.imread(image_path); img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB); img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img = img.astype(np.float32) / 255.0; img = (img - IMAGENET_MEAN) / IMAGENET_STD
        return img
    except Exception: return np.zeros((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)

# *** MODIFIED Text Encoding to include Missing/Garbage Aug ***
def _encode_augmented_text_py(text_tensor, augment_flag_tensor):
    text = text_tensor.numpy().decode('utf-8')
    should_augment = augment_flag_tensor.numpy() # Get boolean flag

    final_text = text # Start with original
    if should_augment and random.random() < APPLY_TEXT_MOD_PROB:
        final_text = augment_text_missing_garbage(text) # Apply new augmentation strategy

    # --- Encode the final text ---
    processed_text = final_text.strip().upper()[:MAX_TEXT_LENGTH]
    encoded = np.zeros((MAX_TEXT_LENGTH,), dtype=np.int32)
    for i, char in enumerate(processed_text):
        encoded[i] = char_map_global.get(char, 0)
    padded_encoded = pad_sequences([encoded], maxlen=MAX_TEXT_LENGTH, padding='post', truncating='post')[0]
    return padded_encoded

@tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.string)])
def tf_load_preprocess_image(image_path_tensor):
    image = tf.py_function(func=_load_and_preprocess_image_py, inp=[image_path_tensor], Tout=tf.float32)
    image.set_shape((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)); return image

# Wrapper for the modified text function
@tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.string), tf.TensorSpec(shape=(), dtype=tf.bool)])
def tf_encode_augmented_text(text_tensor, augment_flag_tensor):
    encoded_text = tf.py_function(
        func=_encode_augmented_text_py,
        inp=[text_tensor, augment_flag_tensor], # Pass flag
        Tout=tf.int32
    )
    encoded_text.set_shape((MAX_TEXT_LENGTH,)); return encoded_text

# Modified dataset creation
def create_tf_dataset(image_paths, texts, labels, is_training=False):
    """Creates tf.data.Dataset, applying text augmentation if training."""
    labels_tensor = tf.cast(labels, dtype=tf.int32)
    dataset = tf.data.Dataset.from_tensor_slices(((image_paths, texts), labels_tensor))

    def load_and_preprocess(inputs, label):
        image_path, text = inputs
        processed_image = tf_load_preprocess_image(image_path)
        # Encode text, passing the boolean flag for augmentation
        encoded_text = tf_encode_augmented_text(text, tf.constant(is_training, dtype=tf.bool))
        one_hot_label = tf.one_hot(label, depth=NUM_CLASSES)
        return (processed_image, encoded_text), one_hot_label

    dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset

# *** MODIFIED Model Building Function for Fine-tuning ***
def build_finetuned_model(num_classes, max_text_len, vocab_size, embedding_dim):
    """Builds the model with base CNN layers unfrozen for fine-tuning."""
    from tensorflow.keras import layers # Ensure import

    # --- Image Input Branch ---
    image_input = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), name="image_input")
    # Load base model
    base_model = tf.keras.applications.MobileNetV2(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
                                                   include_top=False, weights='imagenet') # Load with weights

    # --- Unfreeze Layers for Fine-tuning ---
    base_model.trainable = True # Unfreeze the entire base model first
    # Find the layer index to fine-tune from
    fine_tune_from_layer_name = FINE_TUNE_AT_BLOCK
    fine_tune_at_index = -1
    for i, layer in enumerate(base_model.layers):
        if layer.name == fine_tune_from_layer_name:
            fine_tune_at_index = i
            break

    if fine_tune_at_index != -1:
        print(f"Fine-tuning MobileNetV2 from layer: '{fine_tune_from_layer_name}' (index {fine_tune_at_index})")
        # Freeze all layers before the fine-tune layer
        for layer in base_model.layers[:fine_tune_at_index]:
            layer.trainable = False
    else:
        print(f"Warning: Fine-tune layer '{fine_tune_from_layer_name}' not found. Fine-tuning full base model (might require very low LR).")
    # ------------------------------------

    x = base_model(image_input, training=False) # Run base model in inference mode for BN layers when fine-tuning
                                                # unless you specifically want to train BN layers too.
                                                # Using training=False helps stabilize fine-tuning.

    # --- Top Layers for Image Branch ---
    x = layers.GlobalAveragePooling2D(name="image_pooling")(x)
    # x = layers.BatchNormalization()(x) # Optional: Re-add BN after pooling if needed
    x = layers.Dropout(0.3, name="image_dropout")(x) # Keep dropout
    image_features = layers.Dense(128, activation='relu', name="image_features")(x)

    # --- Text Input Branch (Same as before) ---
    text_input = layers.Input(shape=(max_text_len,), name="text_input", dtype=tf.int32)
    t = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True, name="text_embedding")(text_input)
    t = layers.GRU(64, return_sequences=False, name="text_gru", reset_after=False)(t)
    t = layers.BatchNormalization()(t); t = layers.Dropout(0.3, name="text_dropout")(t)
    text_features = layers.Dense(64, activation='relu', name="text_features")(t)

    # --- Fusion (Same as before) ---
    fused = layers.Concatenate(name="fusion")([image_features, text_features])
    fused = layers.BatchNormalization()(fused); fused = layers.Dense(64, activation='relu', name="fusion_dense")(fused)
    fused = layers.Dropout(0.5, name="fusion_dropout")(fused)
    output = layers.Dense(num_classes, activation='softmax', name="output")(fused)

    model = keras.Model(inputs=[image_input, text_input], outputs=output, name="MM_CMDS_FineTuned_MissingTxt")
    return model

# (Keep plot_training_history function)
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
#                           --- Main Execution ---
# =============================================================================
def main():
    global char_map_global, text_vocab_size_global
    print("--- Running Training (FineTune + MissingText Aug + Weights - CPU) ---")
    print(f"Base Path: {BASE_PATH}")
    if not os.path.exists(BASE_PATH): return

    print("\n" + "="*30); print("IMPORTANT: Ensure CLEANED train/valid data!"); print("="*30 + "\n"); time.sleep(1)
    check_gpu_availability() # Check GPU status

    # --- Load Data ---
    print("\n--- Phase 1: Loading Data ---")
    train_image_paths, train_texts, train_labels_raw = load_data_from_processed_csv(TRAIN_CSV_OCR, TRAIN_DIR)
    valid_image_paths, valid_texts, valid_labels_raw = load_data_from_processed_csv(VALID_CSV_OCR, VALID_DIR)
    test_image_paths, test_texts, test_labels_raw = [], [], []
    if os.path.exists(TEST_CSV_OCR): test_image_paths, test_texts, test_labels_raw = load_data_from_processed_csv(TEST_CSV_OCR, TEST_DIR)
    if not train_image_paths or not valid_image_paths: print("\nError loading data."); return

    # --- Data Prep ---
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

    print("\nBuilding tf.data Pipelines (with missing/garbage text augmentation)...");
    train_dataset = create_tf_dataset(train_image_paths, train_texts, train_labels_raw, is_training=True)
    train_dataset = train_dataset.shuffle(buffer_size=max(1000, len(train_image_paths)//4)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    valid_dataset = create_tf_dataset(valid_image_paths, valid_texts, valid_labels_raw, is_training=False)
    valid_dataset = valid_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_dataset = None
    if test_image_paths:
        test_dataset = create_tf_dataset(test_image_paths, test_texts, test_labels_raw, is_training=False)
        test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        print("Train, Validation, and Test datasets created.")
    else: print("Train and Validation datasets created (No test data).")

    # --- Phase 3: Model Definition & Compilation for Fine-tuning ---
    print("\n--- Phase 3: Model Definition & Compile for Fine-tuning ---")
    # Build the fine-tunable model
    model = build_finetuned_model(NUM_CLASSES, MAX_TEXT_LENGTH, text_vocab_size_global, TEXT_EMBEDDING_DIM)
    model.summary() # Check trainable params - should be higher now

    # Compile with a LOW learning rate
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), # Use low LR
                  loss='categorical_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision(name='precision'),
                  tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.AUC(name='auc')])
    print(f"Model Compiled with ADAM optimizer (LR={LEARNING_RATE}).")

    # --- Phase 4: Model Training ---
    print("\n--- Phase 4: Model Training (Fine-tuning + Aug + Weights - CPU) ---")
    # Adjust patience if needed, monitor val_loss or val_auc
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1) # Less aggressive factor maybe
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
    print("\n--- Phase 5: Model Evaluation ---")
    print(f"Loading best model based on val_auc from: {MODEL_CHECKPOINT_PATH}")
    if os.path.exists(MODEL_CHECKPOINT_PATH):
        try:
            # Load H5 model
            best_model = keras.models.load_model(MODEL_CHECKPOINT_PATH)
            print("Best model loaded successfully.")
            # ... (Rest of evaluation: validation, test, classification report, confusion matrix) ...
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
                    print("  Extracting true labels..."); y_true_indices = test_labels_raw # Use original labels
                    if len(y_true_indices) == len(y_pred_indices):
                         print("\nClassification Report (Test Set):")
                         report = classification_report(y_true_indices, y_pred_indices, target_names=CLASS_NAMES, digits=4)
                         print(report)
                         print("\nConfusion Matrix (Test Set):")
                         cm = confusion_matrix(y_true_indices, y_pred_indices); print(cm)
                         plt.figure(figsize=(6, 5)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
                         plt.xlabel('Predicted Label'); plt.ylabel('True Label'); plt.title('Confusion Matrix - FineTune+MissTxt CPU')
                         plt.savefig(CM_SAVE_PATH); print(f"Confusion matrix plot saved to: {CM_SAVE_PATH}")
                    else: print(f"Error: Label count mismatch")
                except ImportError: print("Warning: scikit-learn or seaborn not installed.")
                except Exception as report_err: print(f"Error during detailed report generation: {report_err}")
            else: print("\nNo test set available for final evaluation.")

        except Exception as e:
            print(f"Error loading or evaluating the best model: {e}")
            print("--- Traceback ---"); traceback.print_exc(); print("-----------------")
    else: print(f"Model checkpoint file not found at {MODEL_CHECKPOINT_PATH}.")
    print("\n--- Script Execution Complete ---")

if __name__ == "__main__":
    main()