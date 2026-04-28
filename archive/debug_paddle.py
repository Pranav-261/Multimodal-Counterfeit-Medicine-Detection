# debug_paddle.py

from paddleocr import PaddleOCR
import cv2
import os

print("--- Starting PaddleOCR Debug Script ---")

# --- USER ACTION: Set the path to ONE image you expect to have text ---
# Use an image that is clear, like one of the medicine boxes.
IMAGE_PATH_TO_TEST = r"Caro_Laptop_Files\test\images39_jpg.rf.27aec12841bf6c3360fcbd09d303b81b.jpg" # Example, CHANGE THIS
# Make sure this file exists!

# Check if the image exists before proceeding
if not os.path.exists(IMAGE_PATH_TO_TEST):
    print(f"FATAL ERROR: Test image not found at '{IMAGE_PATH_TO_TEST}'")
    exit()

# --- Initialize PaddleOCR ---
# Using the new suggested parameter name 'use_angle_cls' might be replaced by 'det_text_angle'
# Or just rely on the default behavior. Let's try the most basic initialization first.
try:
    print("Attempting to initialize PaddleOCR...")
    # Try with minimal arguments first
    ocr_engine = PaddleOCR(lang='en', use_angle_cls=True) # use_gpu defaults to False if CPU-only paddlepaddle is installed
    print("PaddleOCR initialized successfully.")
except Exception as e:
    print(f"FATAL ERROR: Failed to initialize PaddleOCR: {e}")
    exit()

# --- Run OCR on the single image ---
print(f"\n--- Running OCR on: {IMAGE_PATH_TO_TEST} ---")
try:
    # Most robust way: load image with OpenCV, pass the NumPy array
    img_cv = cv2.imread(IMAGE_PATH_TO_TEST)
    if img_cv is None:
        print("FATAL ERROR: OpenCV could not read the image. Check the path and file integrity.")
        exit()

    # --- THE CRITICAL CALL ---
    # The DeprecationWarning says to use .predict(), so let's use that.
    # The .predict() method is the intended replacement for .ocr()
    result = ocr_engine.predict(img_cv)
    # --- END OF CRITICAL CALL ---

    print("\n--- RAW OCR RESULT ---")
    print(result)
    print("--------------------")

    # --- Process the result ---
    ocr_text = ""
    if result and result[0] is not None:
        text_fragments = [line[1][0] for line in result[0] if line and len(line) == 2]
        ocr_text = " ".join(text_fragments).strip()
    
    print("\n--- FINAL PROCESSED TEXT ---")
    if ocr_text:
        print(f"SUCCESS! Extracted Text: '{ocr_text}'")
    else:
        print("FAILURE: No text was extracted. The raw result was empty or in an unexpected format.")

except Exception as e:
    print(f"An error occurred during the OCR process: {e}")
    import traceback
    traceback.print_exc()