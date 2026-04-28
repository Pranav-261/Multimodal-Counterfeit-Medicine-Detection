# -- coding: utf-8 --
"""
train_medicine_model.py

Builds, trains, and evaluates a multi-modal deep learning model for
counterfeit medicine detection using image processing and OCR.

Requirements:
  - tensorflow
  - opencv-python-headless
  - easyocr
  - pandas
  - numpy
  - tqdm
  - matplotlib

Installation (in a virtual environment):
  pip install tensorflow opencv-python-headless easyocr pandas numpy tqdm matplotlib

GPU Setup (Optional but Recommended):
  - Install compatible NVIDIA drivers.
  - Install compatible CUDA Toolkit and cuDNN versions.
  - See: https://www.tensorflow.org/install/gpu
"""
import torch


#import tensorflow as tf
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

from tensorflow import keras
from tensorflow.keras import layers
import cv2 # OpenCV for image processing
import easyocr # For OCR (used only in pre-processing now)
import numpy as np
import pandas as pd # To read/write CSV files
import os
import math # For calculating steps per epoch
from tqdm import tqdm # Use standard tqdm for scripts
import time
import matplotlib.pyplot as plt

# =============================================================================
#                           --- Configuration ---
# =============================================================================

# !!! USER ACTION REQUIRED: Set the base path to your data directory !!!
# This directory should contain 'train', 'valid', and 'test' subfolders.
BASE_PATH = r"Caro_Laptop_Files" # e.g., "/home/user/datasets/medicine_data"
# =============================================================================

IMG_WIDTH = 224
IMG_HEIGHT = 224
IMG_CHANNELS = 3
BATCH_SIZE = 32
EPOCHS = 20 # Adjust as needed based on performance
NUM_CLASSES = 2 # 0 = Counterfeit, 1 = Authentic
MAX_TEXT_LENGTH = 50 # Max characters to consider from OCR
TEXT_EMBEDDING_DIM = 16
LEARNING_RATE = 1e-4

# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# --- Paths ---
TRAIN_DIR = os.path.join(BASE_PATH, "train")
VALID_DIR = os.path.join(BASE_PATH, "valid")
TEST_DIR = os.path.join(BASE_PATH, "test")

# Paths for OCR-processed CSVs (will be created if they don't exist)
TRAIN_CSV_OCR = os.path.join(TRAIN_DIR, "_classes_with_ocr.csv")
VALID_CSV_OCR = os.path.join(VALID_DIR, "_classes_with_ocr.csv")
TEST_CSV_OCR = os.path.join(TEST_DIR, "_classes_with_ocr.csv")

# Path for saving the best model
MODEL_CHECKPOINT_FILENAME = "mm_cmds_model_final.keras_v2"
MODEL_CHECKPOINT_PATH = os.path.join(BASE_PATH, MODEL_CHECKPOINT_FILENAME)

# Path for saving the training history plot
PLOT_FILENAME = "training_history.png"
PLOT_PATH = os.path.join(BASE_PATH, PLOT_FILENAME)


# =============================================================================
#                           --- Helper Functions ---
# =============================================================================

def check_gpu_availability():
    """Checks for GPU and sets memory growth."""
    print("\n--- Checking Hardware ---")
    print(f"TensorFlow Version: {tf.__version__}")
    gpu_devices = tf.config.list_physical_devices('GPU')
    if gpu_devices:
        print(f"GPU Devices Detected: {gpu_devices}")
        try:
            # Setting memory growth is important to prevent TF from allocating all GPU memory at once
            for gpu in gpu_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU Memory Growth Enabled")
        except RuntimeError as e:
            print(f"Warning: Could not set memory growth (might be already set or other issue): {e}")
    else:
        print("No GPU detected by TensorFlow. Training will run on CPU (can be very slow).")
        print("Ensure NVIDIA drivers, CUDA, and cuDNN are installed correctly for GPU usage.")
    return bool(gpu_devices)

def find_original_csv(dir_path, base_name_hint):
    """Finds the original _classes.csv or [hint]_classes.csv file."""
    default_csv = os.path.join(dir_path, "_classes.csv")
    specific_csv = os.path.join(dir_path, f"{base_name_hint}_classes.csv")
    if os.path.exists(default_csv):
        print(f"Found original CSV: {default_csv}")
        return default_csv
    elif os.path.exists(specific_csv):
        print(f"Found original CSV: {specific_csv}")
        return specific_csv
    else:
        print(f"X Warning: Missing original CSV in {dir_path} (looked for _classes.csv and {base_name_hint}_classes.csv)")
        return None

def preprocess_ocr_and_update_csv(original_csv_path, image_dir, output_csv_path, ocr_reader_instance):
    """
    Reads an original CSV, performs OCR on each image,
    and saves a new CSV with an 'extracted_text' column.
    """
    if not ocr_reader_instance:
        print("OCR Reader not available. Skipping OCR pre-processing.")
        return False
    if not original_csv_path or not os.path.exists(original_csv_path):
        print(f"Original CSV not found at {original_csv_path}. Skipping.")
        return False

    print(f"\nProcessing OCR for {original_csv_path}...")
    try:
        df = pd.read_csv(original_csv_path)
        df.iloc[:, 0] = df.iloc[:, 0].str.strip() # Clean filenames
        if 'filename' not in df.columns:
             df.rename(columns={df.columns[0]: 'filename'}, inplace=True)
        if 'label' not in df.columns and len(df.columns) > 1:
             df.rename(columns={df.columns[1]: 'label'}, inplace=True)
    except Exception as e:
        print(f"Error reading CSV {original_csv_path}: {e}")
        return False

    extracted_texts = []
    total_images = len(df)
    start_time = time.time()
    num_ocr_errors = 0
    num_file_not_found = 0

    # Use standard tqdm for script compatibility
    for idx, row in tqdm(df.iterrows(), total=total_images, desc=f"OCR ({os.path.basename(image_dir)})"):
        img_filename = row['filename']
        img_path_base = os.path.join(image_dir, img_filename)
        text = "" # Default to empty string
        actual_img_path = None

        # Handle potential missing extensions and check existence
        if not os.path.splitext(img_path_base)[1]:
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                potential_path = img_path_base + ext
                if os.path.exists(potential_path):
                    actual_img_path = potential_path
                    break
        elif os.path.exists(img_path_base):
            actual_img_path = img_path_base

        if not actual_img_path:
            num_file_not_found += 1
            extracted_texts.append(text) # Append empty text if image missing
            continue

        # Perform OCR
        try:
            img_cv = cv2.imread(actual_img_path)
            if img_cv is None:
                 num_ocr_errors += 1
                 extracted_texts.append(text)
                 continue

            results = ocr_reader_instance.readtext(actual_img_path)
            extracted_text = " ".join([res[1] for res in results])
            text = extracted_text.strip().upper() # Normalize text
        except Exception as e:
            # Optional: Log detailed OCR errors if needed
            # print(f"Warning: Error during OCR for {actual_img_path}: {e}")
            num_ocr_errors += 1
            extracted_texts.append(text) # Append empty string on error

        extracted_texts.append(text)

    df['extracted_text'] = extracted_texts
    df.to_csv(output_csv_path, index=False)
    end_time = time.time()
    print(f"Finished OCR for {os.path.basename(image_dir)}. Saved to {output_csv_path}.")
    if num_file_not_found > 0: print(f"  - Images not found: {num_file_not_found}")
    if num_ocr_errors > 0: print(f"  - OCR/Read errors: {num_ocr_errors}")
    print(f"  - Time taken: {end_time - start_time:.2f} seconds.")
    return True

def load_data_from_processed_csv(csv_path, image_dir):
    """Loads image paths, pre-extracted texts, and labels from the OCR-processed CSV."""
    image_paths = []
    texts = []
    labels = []

    if not os.path.exists(csv_path):
        print(f"Error: Processed CSV file not found at {csv_path}")
        return [], [], []

    try:
        df = pd.read_csv(csv_path)
        if not all(col in df.columns for col in ['filename', 'label', 'extracted_text']):
             print(f"Error: CSV {csv_path} is missing required columns ('filename', 'label', 'extracted_text').")
             return [], [], []

        print(f"Loading data from {csv_path}...")
        df['extracted_text'] = df['extracted_text'].fillna('') # Ensure NaNs become empty strings

        num_skipped = 0
        for idx, row in df.iterrows():
            img_filename = row['filename']
            label = row['label']
            text = str(row['extracted_text'])
            img_path_base = os.path.join(image_dir, img_filename)
            actual_img_path = None

            # Verify image existence again (important in case files were moved/deleted after OCR)
            if not os.path.splitext(img_path_base)[1]:
                 for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                     potential_path = img_path_base + ext
                     if os.path.exists(potential_path):
                         actual_img_path = potential_path
                         break
            elif os.path.exists(img_path_base):
                actual_img_path = img_path_base

            if actual_img_path:
                image_paths.append(actual_img_path)
                texts.append(text)
                labels.append(1 if int(label) == 1 else 0)
            else:
                num_skipped += 1
                continue

        print(f"Loaded {len(image_paths)} samples from {image_dir}. Skipped {num_skipped} due to missing image files.")
        return image_paths, texts, labels

    except Exception as e:
        print(f"Error reading or processing data from {csv_path}: {e}")
        return [], [], []

# --- TF Data Pipeline Functions ---

# Needs access to char_map (global or passed) - using global for simplicity here
# Defined later in the main() function before use
char_map_global = {}

def _load_and_preprocess_image_py(image_path_tensor):
    image_path_bytes = image_path_tensor.numpy()
    image_path = image_path_bytes.decode('utf-8')
    try:
        img = cv2.imread(image_path)
        if img is None:
            return np.zeros((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img = img.astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        return img
    except Exception as e:
        return np.zeros((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)

def _encode_text_py(text_tensor):
    text_bytes = text_tensor.numpy()
    text = text_bytes.decode('utf-8')
    encoded = np.zeros((MAX_TEXT_LENGTH,), dtype=np.int32)
    text = text[:MAX_TEXT_LENGTH]
    for i, char in enumerate(text):
        encoded[i] = char_map_global.get(char, 0) # Use global char_map, 0 for unknown/padding
    return encoded

@tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.string)])
def tf_load_preprocess_image(image_path_tensor):
    image = tf.py_function(
        func=_load_and_preprocess_image_py, inp=[image_path_tensor], Tout=tf.float32
    )
    image.set_shape((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    return image

@tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.string)])
def tf_encode_text(text_tensor):
    encoded_text = tf.py_function(
        func=_encode_text_py, inp=[text_tensor], Tout=tf.int32
    )
    encoded_text.set_shape((MAX_TEXT_LENGTH,))
    return encoded_text

def create_tf_dataset(image_paths, texts, labels):
    """Creates a tf.data.Dataset from image paths, texts, and labels."""
    labels = tf.cast(labels, dtype=tf.int32)
    dataset = tf.data.Dataset.from_tensor_slices(((image_paths, texts), labels))

    def load_and_preprocess(inputs, label):
        image_path, text = inputs
        processed_image = tf_load_preprocess_image(image_path)
        encoded_text = tf_encode_text(text)
        one_hot_label = tf.one_hot(label, depth=NUM_CLASSES)
        return (processed_image, encoded_text), one_hot_label

    dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset

# --- Model Building Function ---

def build_mm_cmds_model(num_classes, max_text_len, vocab_size, embedding_dim):
    """Builds the multi-modal Keras model (MobileNetV2 + GRU)."""
    # --- Image Input Branch ---
    image_input = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), name="image_input")
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
        include_top=False,
        weights='imagenet',
        input_tensor=image_input
    )
    base_model.trainable = False # Freeze base model initially
    x = base_model.output
    x = layers.GlobalAveragePooling2D(name="image_pooling")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3, name="image_dropout")(x)
    image_features = layers.Dense(128, activation='relu', name="image_features")(x)

    # --- Text Input Branch ---
    text_input = layers.Input(shape=(max_text_len,), name="text_input", dtype=tf.int32)
    t = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        mask_zero=True, # Important for handling padding
        name="text_embedding"
    )(text_input)
    # Use standard GRU kernel to avoid potential cuDNN issues
    t = layers.GRU(
        64,
        return_sequences=False,
        name="text_gru",
        reset_after=False # Force non-CuDNN kernel implementation
    )(t)
    t = layers.BatchNormalization()(t)
    t = layers.Dropout(0.3, name="text_dropout")(t)
    text_features = layers.Dense(64, activation='relu', name="text_features")(t)

    # --- Fusion ---
    fused = layers.Concatenate(name="fusion")([image_features, text_features])
    fused = layers.BatchNormalization()(fused)
    fused = layers.Dense(64, activation='relu', name="fusion_dense")(fused)
    fused = layers.Dropout(0.5, name="fusion_dropout")(fused)

    # --- Output Head ---
    output = layers.Dense(num_classes, activation='softmax', name="output")(fused)

    # --- Build the Model ---
    model = keras.Model(inputs=[image_input, text_input], outputs=output, name="MM_CMDS_GRU_StdKernel")
    return model

def plot_training_history(history, save_path):
    """Plots training & validation accuracy and loss."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc)) # Use actual number of epochs run

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\nTraining history plot saved to: {save_path}")
    # plt.show() # Optional: show plot if running interactively

# =============================================================================
#                           --- Main Execution ---
# =============================================================================

def main():
    print("Main execution started")
    """Main function to run the complete pipeline."""
    global char_map_global # Allow modification of the global char_map

    if not os.path.exists(BASE_PATH):
        print(f"Error: BASE_PATH does not exist: {BASE_PATH}")
        print("Please update the BASE_PATH variable in the script.")
        return

    # Check hardware
    has_gpu = check_gpu_availability()

    # --- Phase 1: OCR Pre-processing ---
    print("\n--- Phase 1: OCR Pre-processing ---")

    # Find original CSVs
    orig_train_csv = find_original_csv(TRAIN_DIR, "train")
    orig_valid_csv = find_original_csv(VALID_DIR, "valid")
    orig_test_csv = find_original_csv(TEST_DIR, "test")

    # Initialize OCR reader
    print("Initializing EasyOCR Reader...")
    ocr_reader = None
    try:
        ocr_reader = easyocr.Reader(['en'], gpu=has_gpu) # Use GPU if TF found one
        print("EasyOCR Reader initialized.")
    except Exception as e:
        print(f"Error initializing EasyOCR Reader: {e}. OCR pre-processing will be skipped.")
        # Check if pre-processed files already exist, otherwise exit
        if not (os.path.exists(TRAIN_CSV_OCR) and os.path.exists(VALID_CSV_OCR)):
            print("Error: OCR failed, and pre-processed CSVs not found. Cannot proceed.")
            return

    # Run OCR if needed
    FORCE_OCR_RERUN = False # Set to True to always re-run OCR
    if ocr_reader:
        if not os.path.exists(TRAIN_CSV_OCR) or FORCE_OCR_RERUN:
            if not preprocess_ocr_and_update_csv(orig_train_csv, TRAIN_DIR, TRAIN_CSV_OCR, ocr_reader): return # Exit if error
        else:
            print(f"Found existing OCR file: {TRAIN_CSV_OCR}, skipping OCR run.")

        if not os.path.exists(VALID_CSV_OCR) or FORCE_OCR_RERUN:
             if not preprocess_ocr_and_update_csv(orig_valid_csv, VALID_DIR, VALID_CSV_OCR, ocr_reader): return
        else:
            print(f"Found existing OCR file: {VALID_CSV_OCR}, skipping OCR run.")

        if orig_test_csv:
            if not os.path.exists(TEST_CSV_OCR) or FORCE_OCR_RERUN:
                if not preprocess_ocr_and_update_csv(orig_test_csv, TEST_DIR, TEST_CSV_OCR, ocr_reader): return
            else:
                print(f"Found existing OCR file: {TEST_CSV_OCR}, skipping OCR run.")
        else:
             print("No original test CSV found, skipping test set OCR.")

    # --- Phase 2: Data Loading and Preparation ---
    print("\n--- Phase 2: Data Loading and Preparation ---")
    train_image_paths, train_texts, train_labels = load_data_from_processed_csv(TRAIN_CSV_OCR, TRAIN_DIR)
    valid_image_paths, valid_texts, valid_labels = load_data_from_processed_csv(VALID_CSV_OCR, VALID_DIR)
    test_image_paths, test_texts, test_labels = [], [], []
    if os.path.exists(TEST_CSV_OCR):
        test_image_paths, test_texts, test_labels = load_data_from_processed_csv(TEST_CSV_OCR, TEST_DIR)

    if not train_image_paths or not valid_image_paths:
         print("\nError: Failed to load data for training or validation set. Exiting.")
         return

    # Build Character Map
    print("Building character map from training text...")
    all_train_text = "".join(train_texts)
    unique_chars = sorted(list(set(all_train_text)))
    char_map_global = {char: i+1 for i, char in enumerate(unique_chars)} # 0 for padding
    text_vocab_size = len(char_map_global) + 1
    print(f"Character map built. Vocab size: {text_vocab_size}")

    # Create Datasets
    print("\nBuilding tf.data Pipelines...")
    train_dataset = create_tf_dataset(train_image_paths, train_texts, train_labels)
    train_dataset = train_dataset.shuffle(buffer_size=max(1000, len(train_image_paths) // 4)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    valid_dataset = create_tf_dataset(valid_image_paths, valid_texts, valid_labels)
    valid_dataset = valid_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    test_dataset = None
    if test_image_paths:
        test_dataset = create_tf_dataset(test_image_paths, test_texts, test_labels)
        test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        print("Train, Validation, and Test datasets created.")
    else:
        print("Train and Validation datasets created (No test data).")

    # --- Phase 3: Model Definition ---
    print("\n--- Phase 3: Model Definition ---")
    # Build model (will try GPU first if available, uses standard GRU kernel)
    model = build_mm_cmds_model(
        num_classes=NUM_CLASSES,
        max_text_len=MAX_TEXT_LENGTH,
        vocab_size=text_vocab_size,
        embedding_dim=TEXT_EMBEDDING_DIM
    )
    model.summary()

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy',
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall'),
                 tf.keras.metrics.AUC(name='auc')]
    )
    print("Model Compiled.")

    # --- Phase 4: Model Training ---
    print("\n--- Phase 4: Model Training ---")

    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7, verbose=1)
    print(f"Setting checkpoint path to: {MODEL_CHECKPOINT_PATH}")
    checkpoint = tf.keras.callbacks.ModelCheckpoint(MODEL_CHECKPOINT_PATH, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

    print(f"Training samples: {len(train_image_paths)}")
    print(f"Validation samples: {len(valid_image_paths)}")
    print(f"Batch size: {BATCH_SIZE}, Target Epochs: {EPOCHS}")

    print("\nStarting training...")
    start_train_time = time.time()
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=valid_dataset,
        callbacks=[early_stopping, reduce_lr, checkpoint],
        verbose=1
    )
    end_train_time = time.time()
    print(f"Training finished. Time taken: {end_train_time - start_train_time:.2f} seconds.")

    # Plot history
    if history and history.history:
        plot_training_history(history, PLOT_PATH)

    # --- Phase 5: Evaluation ---
    print("\n--- Phase 5: Model Evaluation ---")
    print(f"Loading best model from: {MODEL_CHECKPOINT_PATH}")
    if os.path.exists(MODEL_CHECKPOINT_PATH):
        try:
            # Load the best model saved during training
            best_model = keras.models.load_model(MODEL_CHECKPOINT_PATH)
            print("Best model loaded successfully.")

            print("\nEvaluating on Validation Set:")
            val_results = best_model.evaluate(valid_dataset, verbose=0)
            print(f"  Validation Loss: {val_results[0]:.4f}")
            print(f"  Validation Accuracy: {val_results[1]:.4f}")
            print(f"  Validation Precision: {val_results[2]:.4f}")
            print(f"  Validation Recall: {val_results[3]:.4f}")
            print(f"  Validation AUC: {val_results[4]:.4f}")

            if test_dataset:
                print("\nEvaluating on Test Set:")
                test_results = best_model.evaluate(test_dataset, verbose=0)
                print(f"  Test Loss: {test_results[0]:.4f}")
                print(f"  Test Accuracy: {test_results[1]:.4f}")
                print(f"  Test Precision: {test_results[2]:.4f}")
                print(f"  Test Recall: {test_results[3]:.4f}")
                print(f"  Test AUC: {test_results[4]:.4f}")
            else:
                print("\nNo test set available for final evaluation.")

        except Exception as e:
            print(f"Error loading or evaluating the best model: {e}")
    else:
        print("Model checkpoint file not found. Evaluation skipped.")
        print("Evaluating the final state of the model (might not be the best):")
        try:
            val_results = model.evaluate(valid_dataset, verbose=0) # Evaluate the model currently in memory
            print(f"  Validation Loss: {val_results[0]:.4f}")
            print(f"  Validation Accuracy: {val_results[1]:.4f}")
            print(f"  Validation Precision: {val_results[2]:.4f}")
            print(f"  Validation Recall: {val_results[3]:.4f}")
            print(f"  Validation AUC: {val_results[4]:.4f}")

            if test_dataset:
                test_results = model.evaluate(test_dataset, verbose=0)
                print(f"\n  Test Loss: {test_results[0]:.4f}")
                print(f"  Test Accuracy: {test_results[1]:.4f}")
                print(f"  Test Precision: {test_results[2]:.4f}")
                print(f"  Test Recall: {test_results[3]:.4f}")
                print(f"  Test AUC: {test_results[4]:.4f}")

        except Exception as e:
             print(f"Error evaluating the final model state: {e}")


    print("\n--- Script Execution Complete ---")

# =============================================================================
# Run the main function when the script is executed
# =============================================================================
if __name__ == "__main__":
    main()
