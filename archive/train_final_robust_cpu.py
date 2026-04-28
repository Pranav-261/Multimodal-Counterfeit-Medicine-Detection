# -*- coding: utf-8 -*-
"""
** current **

**CURRENT IN USE** - "mm_cmds_model_finetune_robust_final.h5"

train_final_robust_cpu.py

Combines:
1. Option to re-run OCR on training data with optimized Tesseract settings.
2. Fine-tuning of the visual base model (MobileNetV2).
3. Class weighting.
4. Advanced text augmentation (including missing/garbage text simulation).
5. Runs entirely on CPU.
"""

# --- Force CPU Usage ---
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# -------------------------------------

import tensorflow as tf
from tensorflow import keras
# ... (all other imports: layers, pad_sequences, cv2, np, pd, math, tqdm, etc.) ...
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
import string
import pytesseract

# =============================================================================
#                           --- Configuration ---
# =============================================================================
BASE_PATH = r"Caro_Laptop_Files"
# ... (rest of your Configuration section: IMG_WIDTH, EPOCHS, LEARNING_RATES, etc.) ...
IMG_WIDTH=224; IMG_HEIGHT=224; IMG_CHANNELS=3; BATCH_SIZE=32; EPOCHS=30
NUM_CLASSES=2; MAX_TEXT_LENGTH=50; TEXT_EMBEDDING_DIM=16;
INITIAL_LR = 1e-4; FINE_TUNE_LR = 1e-5
IMAGENET_MEAN=np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD=np.array([0.229, 0.224, 0.225], dtype=np.float32)
CLASS_NAMES = ['Counterfeit', 'Authentic']
APPLY_TEXT_MOD_PROB = 0.35; MISSING_TEXT_PROB = 0.40; GARBAGE_TEXT_PROB = 0.30
CHAR_SWAP_PROB = 0.15; CHAR_DEL_PROB = 0.10; CHAR_INS_PROB = 0.10
OCR_CONFUSABLES = { 'O': '0', '0': 'O', 'I': '1', '1': 'I', 'L': '1', '1': 'L', 'S': '5', '5': 'S', 'B': '8', '8': 'B', 'G': '6', '6': 'G', 'Z': '2', '2': 'Z',}
GARBAGE_CHARS = string.ascii_uppercase + string.digits + string.punctuation + ' '
FINE_TUNE_AT_BLOCK = 'block_13_expand'; FINE_TUNE_EPOCHS = 15
try: pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
except Exception: pass
TESSERACT_CONFIG_TRAIN = '--psm 3'
# =============================================================================

# --- Paths ---
TRAIN_DIR = os.path.join(BASE_PATH, "train")
VALID_DIR = os.path.join(BASE_PATH, "valid")
TEST_DIR = os.path.join(BASE_PATH, "test")

# *** DEFINE ALL CSV PATHS GLOBALLY ***
# Paths for original EasyOCR-processed CSVs (if RERUN_OCR_FOR_TRAINING_DATA = False)
TRAIN_CSV_OCR = os.path.join(TRAIN_DIR, "_classes_with_ocr.csv")
VALID_CSV_OCR = os.path.join(VALID_DIR, "_classes_with_ocr.csv")
TEST_CSV_OCR = os.path.join(TEST_DIR, "_classes_with_ocr.csv")

# Paths for CSVs potentially updated by Tesseract (if RERUN_OCR_FOR_TRAINING_DATA = True)
TRAIN_CSV_OCR_UPDATED = os.path.join(TRAIN_DIR, "_classes_with_tesseract_ocr_final.csv")
VALID_CSV_OCR_UPDATED = os.path.join(VALID_DIR, "_classes_with_tesseract_ocr_final.csv")
TEST_CSV_OCR_UPDATED = os.path.join(TEST_DIR, "_classes_with_tesseract_ocr_final.csv")

# Model and Plot paths (these were already correct)
MODEL_CHECKPOINT_FILENAME = "mm_cmds_model_finetune_robust_final.h5"
MODEL_CHECKPOINT_PATH = os.path.join(BASE_PATH, MODEL_CHECKPOINT_FILENAME)
PLOT_FILENAME = "training_history_finetune_robust_final.png"
PLOT_PATH = os.path.join(BASE_PATH, PLOT_FILENAME)
CM_FILENAME = "confusion_matrix_finetune_robust_final.png"
CM_SAVE_PATH = os.path.join(BASE_PATH, CM_FILENAME)

# --- Option to re-run OCR on dataset with Tesseract ---
RERUN_OCR_FOR_TRAINING_DATA = False # SET TO TRUE IF YOU WANT TO UPDATE CSVs WITH TESSERACT
# =============================================================================

# =============================================================================
#                           --- Helper Functions ---
# =============================================================================
# (All helper functions: check_gpu_availability, ocr_dataset_with_tesseract,
#  load_data_from_processed_csv, text augmentation functions,
#  _load_and_preprocess_image_py, _encode_augmented_text_py,
#  tf_load_preprocess_image, tf_encode_augmented_text,
#  create_tf_dataset, build_finetuned_model, plot_training_history
#  REMAIN THE SAME AS THE PREVIOUS CORRECT VERSION)
#  Ensure 'layers' and 'pad_sequences' are imported at the top of the script.
# ... (Paste all your helper functions here exactly as they were in the last working iteration) ...
# (For brevity, I'm not re-pasting all of them, but ensure they are there in your file)
# Make sure char_map_global and text_vocab_size_global are defined before TF functions use them.
char_map_global = {}
text_vocab_size_global = 1

def check_gpu_availability():
    print("\n--- Checking Hardware ---"); print(f"TensorFlow Version: {tf.__version__}")
    if 'CUDA_VISIBLE_DEVICES' in os.environ and os.environ['CUDA_VISIBLE_DEVICES'] == '-1':
         print("GPU usage is explicitly disabled via CUDA_VISIBLE_DEVICES.")
         print(f"Visible Physical Devices: {tf.config.list_physical_devices()}")
         return False
    gpu_devices = tf.config.list_physical_devices('GPU')
    if gpu_devices: print(f"GPU Devices Detected: {gpu_devices}"); return True
    else: print("No GPU detected by TensorFlow. Using CPU."); return False

def ocr_dataset_with_tesseract(original_csv_path, image_dir, output_csv_path):
    if not os.path.exists(original_csv_path): print(f"Original CSV {original_csv_path} not found for Tesseract OCR run. Skipping."); return False
    try:
        df = pd.read_csv(original_csv_path); df.iloc[:, 0] = df.iloc[:, 0].str.strip()
        if 'filename' not in df.columns: df.rename(columns={df.columns[0]: 'filename'}, inplace=True)
        if 'label' not in df.columns and len(df.columns) > 1: df.rename(columns={df.columns[1]: 'label'}, inplace=True)
    except Exception as e: print(f"Error reading CSV {original_csv_path}: {e}"); return False
    print(f"\nRunning Tesseract OCR for dataset: {os.path.basename(image_dir)}...")
    new_extracted_texts = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Tesseract OCR on {os.path.basename(image_dir)}"):
        img_filename = row['filename']; img_path_base = os.path.join(image_dir, img_filename); actual_img_path = None
        if not os.path.splitext(img_path_base)[1]:
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                potential_path = img_path_base + ext
                if os.path.exists(potential_path): actual_img_path = potential_path; break
        elif os.path.exists(img_path_base): actual_img_path = img_path_base
        ocr_text = ""
        if actual_img_path:
            try:
                pil_img_for_tesseract = Image.open(actual_img_path)
                ocr_text = pytesseract.image_to_string(pil_img_for_tesseract, lang='eng', config=TESSERACT_CONFIG_TRAIN)
                ocr_text = ocr_text.replace('\n', ' ').replace('\f', '').strip()
            except Exception as e: print(f"Warning: Tesseract OCR failed for {img_filename}: {e}")
        new_extracted_texts.append(ocr_text)
    df['extracted_text'] = new_extracted_texts; df.to_csv(output_csv_path, index=False)
    print(f"Tesseract OCR processed data saved to {output_csv_path}"); return True

def load_data_from_processed_csv(csv_path, image_dir):
    image_paths, texts, labels = [], [], []
    if not os.path.exists(csv_path): print(f"Error: CSV {csv_path} not found."); return [], [], []
    try:
        df = pd.read_csv(csv_path); df['extracted_text'] = df['extracted_text'].fillna('')
        print(f"Loading data from {csv_path}...")
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
        print(f"Loaded {len(image_paths)} samples. Skipped {num_skipped}.")
        return image_paths, texts, labels
    except Exception as e: print(f"Error loading {csv_path}: {e}"); return [], [], []

def random_char_swap(text_list): # ... (Same as before)
    if not text_list: return text_list
    idx = random.randint(0, len(text_list)-1); char_to_swap = text_list[idx]
    if char_to_swap in OCR_CONFUSABLES and random.random() < CHAR_SWAP_PROB: text_list[idx] = OCR_CONFUSABLES[char_to_swap]
    return text_list
def random_char_delete(text_list): # ... (Same as before)
    if not text_list: return text_list
    if random.random() < CHAR_DEL_PROB and len(text_list) > 1: idx = random.randint(0, len(text_list)-1); del text_list[idx]
    return text_list
def random_char_insert(text_list): # ... (Same as before)
    if random.random() < CHAR_INS_PROB and len(text_list) < MAX_TEXT_LENGTH:
        idx = random.randint(0, len(text_list)); random_char = random.choice(list(char_map_global.keys()) + [' ','.','-'])
        text_list.insert(idx, random_char)
    return text_list
def generate_garbage_text(length): return ''.join(random.choice(GARBAGE_CHARS) for _ in range(length)) # ... (Same)
def augment_text_missing_garbage(text): # ... (Same as before)
    mod_type = random.random()
    if mod_type < MISSING_TEXT_PROB: return ""
    elif mod_type < MISSING_TEXT_PROB + GARBAGE_TEXT_PROB: return generate_garbage_text(random.randint(5, MAX_TEXT_LENGTH))
    else: text_list = list(text); text_list = random_char_swap(text_list); text_list = random_char_delete(text_list); text_list = random_char_insert(text_list); return "".join(text_list)

def _load_and_preprocess_image_py(image_path_tensor): # ... (Same)
    image_path = image_path_tensor.numpy().decode('utf-8')
    try: img = cv2.imread(image_path); img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB); img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT)); img = img.astype(np.float32) / 255.0; img = (img - IMAGENET_MEAN) / IMAGENET_STD; return img
    except Exception: return np.zeros((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)
def _encode_augmented_text_py(text_tensor, augment_flag_tensor): # ... (Same)
    text = text_tensor.numpy().decode('utf-8'); should_augment = augment_flag_tensor.numpy(); final_text = text
    if should_augment and random.random() < APPLY_TEXT_MOD_PROB: final_text = augment_text_missing_garbage(text)
    processed_text = final_text.strip().upper()[:MAX_TEXT_LENGTH]; encoded = np.zeros((MAX_TEXT_LENGTH,), dtype=np.int32)
    for i, char in enumerate(processed_text): encoded[i] = char_map_global.get(char, 0)
    padded_encoded = pad_sequences([encoded], maxlen=MAX_TEXT_LENGTH, padding='post', truncating='post')[0]
    return padded_encoded

@tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.string)]) # ... (Same)
def tf_load_preprocess_image(image_path_tensor): image = tf.py_function(func=_load_and_preprocess_image_py, inp=[image_path_tensor], Tout=tf.float32); image.set_shape((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)); return image
@tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.string), tf.TensorSpec(shape=(), dtype=tf.bool)]) # ... (Same)
def tf_encode_augmented_text(text_tensor, augment_flag_tensor): encoded_text = tf.py_function(func=_encode_augmented_text_py, inp=[text_tensor, augment_flag_tensor], Tout=tf.int32); encoded_text.set_shape((MAX_TEXT_LENGTH,)); return encoded_text
def create_tf_dataset(image_paths, texts, labels, is_training=False): # ... (Same)
    labels_tensor = tf.cast(labels, dtype=tf.int32); dataset = tf.data.Dataset.from_tensor_slices(((image_paths, texts), labels_tensor))
    def load_and_preprocess(inputs, label):
        image_path, text = inputs; processed_image = tf_load_preprocess_image(image_path)
        encoded_text = tf_encode_augmented_text(text, tf.constant(is_training, dtype=tf.bool))
        one_hot_label = tf.one_hot(label, depth=NUM_CLASSES)
        return (processed_image, encoded_text), one_hot_label
    dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE); return dataset

def build_finetuned_model(num_classes, max_text_len, vocab_size, embedding_dim): # ... (Same)
    from tensorflow.keras import layers
    image_input = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), name="image_input")
    base_model = tf.keras.applications.MobileNetV2(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), include_top=False, weights='imagenet')
    base_model.trainable = True; fine_tune_at_index = -1
    for i, layer in enumerate(base_model.layers):
        if layer.name == FINE_TUNE_AT_BLOCK: fine_tune_at_index = i; break
    if fine_tune_at_index != -1: print(f"Fine-tuning from layer: '{FINE_TUNE_AT_BLOCK}' (index {fine_tune_at_index})"); [setattr(layer, 'trainable', False) for layer in base_model.layers[:fine_tune_at_index]]
    else: print(f"Warning: Fine-tune layer '{FINE_TUNE_AT_BLOCK}' not found. Fine-tuning full base model.")
    x = base_model(image_input, training=False); x = layers.GlobalAveragePooling2D(name="image_pooling")(x)
    x = layers.Dropout(0.3, name="image_dropout")(x); image_features = layers.Dense(128, activation='relu', name="image_features")(x)
    text_input = layers.Input(shape=(max_text_len,), name="text_input", dtype=tf.int32)
    t = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True, name="text_embedding")(text_input)
    t = layers.GRU(64, return_sequences=False, name="text_gru", reset_after=False)(t)
    t = layers.BatchNormalization()(t); t = layers.Dropout(0.3, name="text_dropout")(t); text_features = layers.Dense(64, activation='relu', name="text_features")(t)
    fused = layers.Concatenate(name="fusion")([image_features, text_features]); fused = layers.BatchNormalization()(fused); fused = layers.Dense(64, activation='relu', name="fusion_dense")(fused)
    fused = layers.Dropout(0.5, name="fusion_dropout")(fused); output = layers.Dense(num_classes, activation='softmax', name="output")(fused)
    model = keras.Model(inputs=[image_input, text_input], outputs=output, name="MM_CMDS_FineTuned_Robust"); return model

def plot_training_history(history, save_path):
    # ... (plotting code remains the same) ...
    acc=history.history['accuracy']; val_acc=history.history['val_accuracy']
    loss=history.history['loss']; val_loss=history.history['val_loss']
    epochs_range = range(len(acc)) # Use actual number of epochs run

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.tight_layout() # This should be on its own line
    try:
        plt.savefig(save_path)
        print(f"\nTraining history plot saved to: {save_path}")
    except Exception as e:
        print(f"\nError saving training plot: {e}")
    # plt.show() # Optional: show plot if running interactively
# =============================================================================

# =============================================================================
#                           --- Main Execution ---
# =============================================================================
def main():
    global char_map_global, text_vocab_size_global
    print("--- Running Training (FineTune + AdvancedTextAug + Weights - CPU) ---")
    print(f"Base Path: {BASE_PATH}")
    if not os.path.exists(BASE_PATH):
        print(f"Error: BASE_PATH '{BASE_PATH}' does not exist.")
        return

    print("\n" + "="*30); print("IMPORTANT: Ensure CLEANED train/valid data!"); print("="*30 + "\n"); time.sleep(1)
    check_gpu_availability()

    # --- Define which CSVs to use ---
    if RERUN_OCR_FOR_TRAINING_DATA:
        print("\n--- Re-running OCR on dataset with Tesseract ---")
        orig_train_csv = os.path.join(TRAIN_DIR, "_classes.csv")
        orig_valid_csv = os.path.join(VALID_DIR, "_classes.csv")
        orig_test_csv = os.path.join(TEST_DIR, "_classes.csv")

        # Use globally defined TRAIN_CSV_OCR_UPDATED, etc. for output
        if not ocr_dataset_with_tesseract(orig_train_csv, TRAIN_DIR, TRAIN_CSV_OCR_UPDATED): return
        if not ocr_dataset_with_tesseract(orig_valid_csv, VALID_DIR, VALID_CSV_OCR_UPDATED): return
        if os.path.exists(orig_test_csv):
            if not ocr_dataset_with_tesseract(orig_test_csv, TEST_DIR, TEST_CSV_OCR_UPDATED): return

        current_train_csv = TRAIN_CSV_OCR_UPDATED
        current_valid_csv = VALID_CSV_OCR_UPDATED
        current_test_csv = TEST_CSV_OCR_UPDATED
    else:
        print("\n--- Using existing OCR-processed CSVs ---")
        # Use the globally defined EasyOCR paths
        current_train_csv = TRAIN_CSV_OCR # This was the missing link
        current_valid_csv = VALID_CSV_OCR
        current_test_csv = TEST_CSV_OCR
    # -----------------------------------------------------------------------

    # --- Load Data ---
    print("\n--- Phase 1b: Loading Data ---")
    # ... (rest of main() function is the same as the previous CORRECTED version) ...
    # (It includes Data Prep, Model Def, Initial Training, Fine-tuning, Evaluation)
    train_image_paths, train_texts, train_labels_raw = load_data_from_processed_csv(current_train_csv, TRAIN_DIR)
    valid_image_paths, valid_texts, valid_labels_raw = load_data_from_processed_csv(current_valid_csv, VALID_DIR)
    test_image_paths, test_texts, test_labels_raw = [], [], []
    if os.path.exists(current_test_csv): test_image_paths, test_texts, test_labels_raw = load_data_from_processed_csv(current_test_csv, TEST_DIR)
    if not train_image_paths or not valid_image_paths: print("\nError loading data. Exiting."); return
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
    print("\n--- Phase 3.1: Initial Training (Frozen Base Model) ---")
    model = build_finetuned_model(NUM_CLASSES, MAX_TEXT_LENGTH, text_vocab_size_global, TEXT_EMBEDDING_DIM)
    base_model_layer = None
    for layer in model.layers:
        if layer.name.startswith('mobilenetv2'): base_model_layer = layer; break
    if base_model_layer: base_model_layer.trainable = False; print(f"Base model '{base_model_layer.name}' frozen.")
    else: print("Warning: Could not find MobileNetV2 layer to freeze.")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=INITIAL_LR), loss='categorical_crossentropy', metrics=['accuracy', 'AUC'])
    print("Initial model compiled for training top layers.")
    initial_epochs_to_run = EPOCHS - FINE_TUNE_EPOCHS
    if initial_epochs_to_run <= 0: initial_epochs_to_run = max(1, EPOCHS // 3)
    print(f"\nTraining top layers for {initial_epochs_to_run} epochs...")
    history_initial = model.fit(train_dataset, epochs=initial_epochs_to_run, validation_data=valid_dataset, class_weight=calculated_class_weights, verbose=1)
    print("\n--- Phase 3.2: Fine-tuning Model ---")
    if base_model_layer:
        base_model_layer.trainable = True; fine_tune_at_index = -1
        for i, layer_in_base in enumerate(base_model_layer.layers):
            if layer_in_base.name == FINE_TUNE_AT_BLOCK: fine_tune_at_index = i; break
        if fine_tune_at_index != -1: print(f"Fine-tuning base model from layer: '{FINE_TUNE_AT_BLOCK}' (idx {fine_tune_at_index})"); [setattr(layer_in_base, 'trainable', False) for layer_in_base in base_model_layer.layers[:fine_tune_at_index]]
        else: print(f"Warning: Fine-tune layer '{FINE_TUNE_AT_BLOCK}' not found. Fine-tuning more of base.")
    else: print("Warning: Base model layer not found for fine-tuning setup.")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_LR), loss='categorical_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.AUC(name='auc')])
    print(f"Model Re-Compiled for Fine-tuning (LR={FINE_TUNE_LR}).")
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1)
    print(f"Checkpoint path: {MODEL_CHECKPOINT_PATH}")
    checkpoint = tf.keras.callbacks.ModelCheckpoint(MODEL_CHECKPOINT_PATH, monitor='val_auc', save_best_only=True, mode='max', verbose=1)
    print(f"\nStarting fine-tuning for {FINE_TUNE_EPOCHS} epochs...")
    start_train_time = time.time()
    start_fine_tune_epoch = 0;
    if history_initial and history_initial.epoch: start_fine_tune_epoch = history_initial.epoch[-1] + 1
    history_finetune = model.fit(train_dataset, epochs=start_fine_tune_epoch + FINE_TUNE_EPOCHS, initial_epoch=start_fine_tune_epoch, validation_data=valid_dataset, callbacks=[early_stopping, reduce_lr, checkpoint], class_weight=calculated_class_weights, verbose=1)
    end_train_time = time.time(); print(f"Fine-tuning finished. Time taken: {end_train_time - start_train_time:.2f} seconds.")
    if history_finetune and history_finetune.history: plot_training_history(history_finetune, PLOT_PATH)
    print("\n--- Phase 5: Model Evaluation ---")
    print(f"Loading best model from: {MODEL_CHECKPOINT_PATH}")
    if os.path.exists(MODEL_CHECKPOINT_PATH):
        try:
            best_model = keras.models.load_model(MODEL_CHECKPOINT_PATH)
            print("Best model loaded successfully.")
            print("\nEvaluating on Validation Set:")
            val_results = best_model.evaluate(valid_dataset, verbose=0, return_dict=True)
            print(f"  Val Loss: {val_results['loss']:.4f}, Acc: {val_results['accuracy']:.4f}, Prec: {val_results['precision']:.4f}, Rec: {val_results['recall']:.4f}, AUC: {val_results['auc']:.4f}")
            if test_dataset:
                print("\nEvaluating on Test Set:")
                test_results = best_model.evaluate(test_dataset, verbose=0, return_dict=True)
                print(f"  Test Loss: {test_results['loss']:.4f}, Acc: {test_results['accuracy']:.4f}, Prec: {test_results['precision']:.4f}, Rec: {test_results['recall']:.4f}, AUC: {test_results['auc']:.4f}")
                print("\nGenerating Detailed Classification Report on Test Set...")
                try:
                    y_pred_probs = best_model.predict(test_dataset); y_pred_indices = np.argmax(y_pred_probs, axis=1)
                    y_true_indices = test_labels_raw
                    if len(y_true_indices) == len(y_pred_indices):
                         report = classification_report(y_true_indices, y_pred_indices, target_names=CLASS_NAMES, digits=4)
                         print("\nClassification Report (Test Set):\n", report)
                         cm = confusion_matrix(y_true_indices, y_pred_indices); print("\nConfusion Matrix (Test Set):\n", cm)
                         plt.figure(figsize=(6, 5)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
                         plt.xlabel('Predicted Label'); plt.ylabel('True Label'); plt.title('CM - FineTune+MissingTxt CPU')
                         plt.savefig(CM_SAVE_PATH); print(f"Confusion matrix plot saved to: {CM_SAVE_PATH}")
                    else: print(f"Error: Label count mismatch")
                except ImportError: print("Warning: scikit-learn or seaborn not installed.")
                except Exception as report_err: print(f"Error in report generation: {report_err}")
        except Exception as e: print(f"Error loading/evaluating: {e}"); traceback.print_exc()
    else: print(f"Model checkpoint not found at {MODEL_CHECKPOINT_PATH}.")
    print("\n--- Script Execution Complete ---")

if __name__ == "__main__":
    main()