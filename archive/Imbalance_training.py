import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2 # for image processing
import easyocr # For OCR 
import numpy as np
import pandas as pd 
import os
import math 
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import zipfile 

BASE_PATH = r"Caro_Laptop_Files" 
IMG_WIDTH = 224
IMG_HEIGHT = 224
IMG_CHANNELS = 3
BATCH_SIZE = 32
EPOCHS = 30 
NUM_CLASSES = 2 # 0 = Counterfeit, 1 = Authentic
MAX_TEXT_LENGTH = 50
TEXT_EMBEDDING_DIM = 16
LEARNING_RATE = 1e-4 

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

TRAIN_DIR = os.path.join(BASE_PATH, "train")
VALID_DIR = os.path.join(BASE_PATH, "valid")
TEST_DIR = os.path.join(BASE_PATH, "test")

TRAIN_CSV_OCR = os.path.join(TRAIN_DIR, "_classes_with_ocr.csv")
VALID_CSV_OCR = os.path.join(VALID_DIR, "_classes_with_ocr.csv")
TEST_CSV_OCR = os.path.join(TEST_DIR, "_classes_with_ocr.csv")

MODEL_CHECKPOINT_FILENAME = "mm_cmds_model_weighted_final.keras"
MODEL_CHECKPOINT_PATH = os.path.join(BASE_PATH, MODEL_CHECKPOINT_FILENAME)

PLOT_FILENAME = "training_history_weighted.png"
PLOT_PATH = os.path.join(BASE_PATH, PLOT_FILENAME)

def check_gpu_availability():
    """Checks for GPU but acknowledges CPU override."""
    print("\n--- Checking Hardware ---")
    print(f"TensorFlow Version: {tf.__version__}")
    if 'CUDA_VISIBLE_DEVICES' in os.environ and os.environ['CUDA_VISIBLE_DEVICES'] == '-1':
         print("GPU usage is explicitly disabled via CUDA_VISIBLE_DEVICES.")
         print(f"Visible Physical Devices: {tf.config.list_physical_devices()}")
         return False # Report no GPU 
    gpu_devices = tf.config.list_physical_devices('GPU')
    if gpu_devices:
        print(f"GPU Devices Detected: {gpu_devices}")
        try:
            for gpu in gpu_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU Memory Growth Enabled")
            return True
        except RuntimeError as e:
            print(f"Warning: Could not set memory growth: {e}")
            return True # GPU exists
    else:
        print("No GPU detected by TensorFlow. Using CPU.")
        return False

def load_data_from_processed_csv(csv_path, image_dir):
    """Loads image paths, pre-extracted texts, and labels from the OCR-processed CSV."""
    image_paths = []
    texts = []
    labels = [] # store 0/1 labels

    if not os.path.exists(csv_path):
        print(f"Error: Processed CSV file not found at {csv_path}")
        return [], [], []

    try:
        df = pd.read_csv(csv_path)
        if not all(col in df.columns for col in ['filename', 'label', 'extracted_text']):
             print(f"Error: CSV {csv_path} is missing required columns ('filename', 'label', 'extracted_text').")
             return [], [], []

        print(f"Loading data from {csv_path}...")
        df['extracted_text'] = df['extracted_text'].fillna('') 

        num_skipped = 0
        for idx, row in df.iterrows():
            img_filename = row['filename']
            label_val = int(row['label']) 
            text = str(row['extracted_text'])
            img_path_base = os.path.join(image_dir, img_filename)
            actual_img_path = None

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
                labels.append(1 if label_val == 1 else 0) 
            else:
                num_skipped += 1
                continue

        print(f"Loaded {len(image_paths)} samples from {image_dir}. Skipped {num_skipped} due to missing image files.")
        return image_paths, texts, labels

    except Exception as e:
        print(f"Error reading or processing data from {csv_path}: {e}")
        return [], [], []

char_map_global = {} 

def _load_and_preprocess_image_py(image_path_tensor):
    image_path_bytes = image_path_tensor.numpy()
    image_path = image_path_bytes.decode('utf-8')
    try:
        img = cv2.imread(image_path)
        if img is None: return np.zeros((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)
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
        encoded[i] = char_map_global.get(char, 0)
    return encoded

@tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.string)])
def tf_load_preprocess_image(image_path_tensor):
    image = tf.py_function(func=_load_and_preprocess_image_py, inp=[image_path_tensor], Tout=tf.float32)
    image.set_shape((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    return image

@tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.string)])
def tf_encode_text(text_tensor):
    encoded_text = tf.py_function(func=_encode_text_py, inp=[text_tensor], Tout=tf.int32)
    encoded_text.set_shape((MAX_TEXT_LENGTH,))
    return encoded_text

def create_tf_dataset(image_paths, texts, labels):
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

def build_mm_cmds_model(num_classes, max_text_len, vocab_size, embedding_dim):
    """Builds the multi-modal Keras model (MobileNetV2 + GRU)."""
    image_input = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), name="image_input")
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
        include_top=False, weights='imagenet', input_tensor=image_input)
    base_model.trainable = False
    x = base_model.output
    x = layers.GlobalAveragePooling2D(name="image_pooling")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3, name="image_dropout")(x)
    image_features = layers.Dense(128, activation='relu', name="image_features")(x)
    text_input = layers.Input(shape=(max_text_len,), name="text_input", dtype=tf.int32)
    t = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True, name="text_embedding")(text_input)
    t = layers.GRU(64, return_sequences=False, name="text_gru", reset_after=False)(t)
    t = layers.BatchNormalization()(t)
    t = layers.Dropout(0.3, name="text_dropout")(t)
    text_features = layers.Dense(64, activation='relu', name="text_features")(t)
    fused = layers.Concatenate(name="fusion")([image_features, text_features])
    fused = layers.BatchNormalization()(fused)
    fused = layers.Dense(64, activation='relu', name="fusion_dense")(fused)
    fused = layers.Dropout(0.5, name="fusion_dropout")(fused)
    output = layers.Dense(num_classes, activation='softmax', name="output")(fused)
    model = keras.Model(inputs=[image_input, text_input], outputs=output, name="MM_CMDS_GRU_StdKernel")
    return model

def plot_training_history(history, save_path):
    """Plots training & validation accuracy and loss."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

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
    plt.tight_layout()
    try:
        plt.savefig(save_path)
        print(f"\nTraining history plot saved to: {save_path}")
    except Exception as e:
        print(f"\nError saving training plot: {e}")
    # plt.show()

def main():
    """Main function to run the complete pipeline with class weighting."""
    global char_map_global 

    print("--- Running Counterfeit Medicine Detection Training (with Class Weighting) ---")
    print(f"Base Path: {BASE_PATH}")
    if not os.path.exists(BASE_PATH):
        print(f"Error: BASE_PATH does not exist: {BASE_PATH}")
        return

    print("\n" + "="*30)
    print("IMPORTANT: Ensure your train/valid datasets have had duplicates removed!")
    print("This script assumes data leakage has been addressed.")
    print("="*30 + "\n")
    time.sleep(2) 

    has_gpu = check_gpu_availability()

    # Phase 1: Load Data
    print("\n--- Phase 1: Loading Data (Assuming OCR is Pre-Processed) ---")
    train_image_paths, train_texts, train_labels_raw = load_data_from_processed_csv(TRAIN_CSV_OCR, TRAIN_DIR)
    valid_image_paths, valid_texts, valid_labels_raw = load_data_from_processed_csv(VALID_CSV_OCR, VALID_DIR)
    test_image_paths, test_texts, test_labels_raw = [], [], []
    if os.path.exists(TEST_CSV_OCR):
        test_image_paths, test_texts, test_labels_raw = load_data_from_processed_csv(TEST_CSV_OCR, TEST_DIR)

    if not train_image_paths or not valid_image_paths:
         print("\nError: Failed to load data for training or validation set. Exiting.")
         return

    # Phase 2: Data Preparation
    print("\n--- Phase 2: Data Preparation ---")
    print("Building character map from training text...")
    all_train_text = "".join(train_texts)
    unique_chars = sorted(list(set(all_train_text)))
    char_map_global = {char: i+1 for i, char in enumerate(unique_chars)}
    text_vocab_size = len(char_map_global) + 1
    print(f"Character map built. Vocab size: {text_vocab_size}")

    print("Calculating class weights for training data...")
    neg_count = sum(1 for label in train_labels_raw if label == 0) # Counterfeit count
    pos_count = len(train_labels_raw) - neg_count # Authentic count
    total_count = len(train_labels_raw)

    calculated_class_weights = {}
    if neg_count > 0 and pos_count > 0:
        # formula: weight = (total / (n_classes * n_samples_class))
        weight_for_0 = (total_count / (2.0 * neg_count))
        weight_for_1 = (total_count / (2.0 * pos_count))
        calculated_class_weights = {0: weight_for_0, 1: weight_for_1}
        print(f"  Counterfeit (0) count: {neg_count}")
        print(f"  Authentic (1) count: {pos_count}")
        print(f"  Calculated Class Weights: {calculated_class_weights}")
    else:
        print("Warning: One class has zero samples in the training data. Cannot calculate weights.")
        calculated_class_weights = {0: 1.0, 1: 1.0} 
        print("Using default equal weights.")

    # Create Datasets
    print("\nBuilding tf.data Pipelines...")
    train_dataset = create_tf_dataset(train_image_paths, train_texts, train_labels_raw)
    train_dataset = train_dataset.shuffle(buffer_size=max(1000, len(train_image_paths) // 4)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    valid_dataset = create_tf_dataset(valid_image_paths, valid_texts, valid_labels_raw)
    valid_dataset = valid_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    test_dataset = None
    if test_image_paths:
        test_dataset = create_tf_dataset(test_image_paths, test_texts, test_labels_raw)
        test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        print("Train, Validation, and Test datasets created.")
    else:
        print("Train and Validation datasets created (No test data).")

    # Phase 3: Model Definition
    print("\n--- Phase 3: Model Definition ---")
    model = build_mm_cmds_model(
        num_classes=NUM_CLASSES,
        max_text_len=MAX_TEXT_LENGTH,
        vocab_size=text_vocab_size,
        embedding_dim=TEXT_EMBEDDING_DIM
    )
    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy',
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall'),
                 tf.keras.metrics.AUC(name='auc')]
    )
    print("Model Compiled.")

    # Phase 4: Model Training (with Class Weighting)
    print("\n--- Phase 4: Model Training (Applying Class Weights) ---")

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1) 
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, min_lr=1e-7, verbose=1) 
    print(f"Setting checkpoint path to: {MODEL_CHECKPOINT_PATH}")
    checkpoint = tf.keras.callbacks.ModelCheckpoint(MODEL_CHECKPOINT_PATH, monitor='val_auc', save_best_only=True, mode='max', verbose=1)

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
        class_weight=calculated_class_weights, 
        verbose=1
    )
    end_train_time = time.time()
    print(f"Training finished. Time taken: {end_train_time - start_train_time:.2f} seconds.")

    if history and history.history:
        plot_training_history(history, PLOT_PATH)

    # Phase 5: Evaluation
    print("\n--- Phase 5: Model Evaluation ---")
    print(f"Loading best model based on val_auc from: {MODEL_CHECKPOINT_PATH}")
    if os.path.exists(MODEL_CHECKPOINT_PATH):
        try:
            best_model = keras.models.load_model(MODEL_CHECKPOINT_PATH)
            print("Best model loaded successfully.")

            print("\nEvaluating on Validation Set:")
            val_results = best_model.evaluate(valid_dataset, verbose=0, return_dict=True)
            print(f"  Validation Loss: {val_results['loss']:.4f}")
            print(f"  Validation Accuracy: {val_results['accuracy']:.4f}")
            print(f"  Validation Precision: {val_results['precision']:.4f}")
            print(f"  Validation Recall: {val_results['recall']:.4f}")
            print(f"  Validation AUC: {val_results['auc']:.4f}")

            if test_dataset:
                print("\nEvaluating on Test Set:")
                test_results = best_model.evaluate(test_dataset, verbose=0, return_dict=True)
                print(f"  Test Loss: {test_results['loss']:.4f}")
                print(f"  Test Accuracy: {test_results['accuracy']:.4f}")
                print(f"  Test Precision: {test_results['precision']:.4f}")
                print(f"  Test Recall: {test_results['recall']:.4f}")
                print(f"  Test AUC: {test_results['auc']:.4f}")

                print("\nGenerating Detailed Classification Report on Test Set...")
                try:
                    from sklearn.metrics import classification_report, confusion_matrix
                    import seaborn as sns

                    print("  Running predictions...")
                    y_pred_probs = best_model.predict(test_dataset)
                    y_pred_indices = np.argmax(y_pred_probs, axis=1)

                    print("  Extracting true labels...")
                    y_true_indices = []
                    for _, labels_batch in test_dataset.unbatch().batch(len(test_labels_raw)): 
                        y_true_indices.extend(np.argmax(labels_batch.numpy(), axis=1)) 
                        break 

                    if len(y_true_indices) == len(y_pred_indices):
                         print("\nClassification Report (Test Set):")
                         target_names = ['Counterfeit (0)', 'Authentic (1)']
                         report = classification_report(y_true_indices, y_pred_indices, target_names=target_names)
                         print(report)

                         print("\nConfusion Matrix (Test Set):")
                         cm = confusion_matrix(y_true_indices, y_pred_indices)
                         print(cm)
                         plt.figure(figsize=(6, 5))
                         sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
                         plt.xlabel('Predicted Label')
                         plt.ylabel('True Label')
                         plt.title('Confusion Matrix - Test Set (Weighted Model)')
                         cm_save_path = os.path.join(BASE_PATH, "confusion_matrix_weighted.png")
                         plt.savefig(cm_save_path)
                         print(f"Confusion matrix plot saved to: {cm_save_path}")
                         # plt.show()
                    else:
                         print(f"Error: Label count mismatch ({len(y_true_indices)} vs {len(y_pred_indices)})")

                except ImportError:
                    print("Warning: scikit-learn or seaborn not installed. Cannot generate detailed report/matrix.")
                    print("Install with: pip install scikit-learn seaborn")
                except Exception as report_err:
                    print(f"Error during detailed report generation: {report_err}")

            else:
                print("\nNo test set available for final evaluation.")

        except Exception as e:
            print(f"Error loading or evaluating the best model: {e}")
    else:
        print("Model checkpoint file not found. Evaluation skipped.")

    print("\n--- Script Execution Complete ---")

if __name__ == "__main__":
    main()