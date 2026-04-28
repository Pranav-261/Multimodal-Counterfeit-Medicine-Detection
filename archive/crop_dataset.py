# crop_dataset.py (v4 - Final, Corrected Logic)

import cv2
import os
import pandas as pd
from tqdm import tqdm

# =============================================================================
#                           --- Configuration ---
# =============================================================================
SOURCE_BASE_PATH = r"Caro_Laptop_Files"
DESTINATION_BASE_PATH = r"Caro_Laptop_Files_CROPPED"
SETS_TO_PROCESS = ["test", "valid", "train"] # Process smaller sets first
# =============================================================================

def interactive_crop_and_save(source_dir, dest_dir):
    """
    Iterates through images, allows interactive cropping OR accepting as-is.
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    original_csv_path = os.path.join(source_dir, "_classes.csv")
    if not os.path.exists(original_csv_path):
        print(f"Error: Original CSV not found at {original_csv_path}. Skipping.")
        return "CONTINUE"

    df = pd.read_csv(original_csv_path)
    if 'filename' not in df.columns: df.rename(columns={df.columns[0]: 'filename'}, inplace=True)
    if 'label' not in df.columns and len(df.columns) > 1: df.rename(columns={df.columns[1]: 'label'}, inplace=True)

    new_data = []
    
    print(f"\n--- Starting interactive cropping for: {source_dir} ---")
    print("INSTRUCTIONS:")
    print("  - To CROP: Drag a box with the mouse, then press 'ENTER' or 'SPACE'.")
    print("  - To ACCEPT AS-IS: DO NOT draw a box, just press 'ENTER' or 'SPACE'.")
    print("  - To SKIP: Press the 'c' key.")
    print("  - To QUIT: Press the 'ESC' key.")

    window_name = "Crop (ENTER/SPACE) | Accept (ENTER/SPACE) | Skip (c) | Quit (ESC)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) # Make it resizable

    for index, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {os.path.basename(source_dir)}"):
        filename = row['filename']
        label = row['label']
        
        # Find the full image path
        img_path_base = os.path.join(source_dir, filename)
        actual_img_path = None
        if not os.path.splitext(img_path_base)[1]:
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                if os.path.exists(img_path_base + ext): actual_img_path = img_path_base + ext; break
        elif os.path.exists(img_path_base): actual_img_path = img_path_base

        if not actual_img_path: print(f"Skipping missing file: {filename}"); continue
        
        image = cv2.imread(actual_img_path)
        if image is None: print(f"Skipping unreadable file: {filename}"); continue

        # Use selectROI and it will block until user acts
        roi = cv2.selectROI(window_name, image, fromCenter=False, showCrosshair=True)
        # waitKey(0) after selectROI captures the key that was used to exit selectROI
        key = cv2.waitKey(0) & 0xFF

        if key == 27: # ESC key to quit immediately
            print("\nQuitting...")
            cv2.destroyAllWindows()
            # Save progress before quitting
            if new_data:
                new_df = pd.DataFrame(new_data)
                new_csv_path = os.path.join(dest_dir, "_classes.csv")
                new_df.to_csv(new_csv_path, index=False)
                print(f"\nProgress CSV saved for {os.path.basename(source_dir)} at: {new_csv_path}")
            return "QUIT"
        
        elif key == ord('c'): # 'c' key to skip
            print(f" -> Skipped image: {filename}")
            continue

        # Check if a box was drawn (width and height are non-zero)
        elif roi[2] > 0 and roi[3] > 0:
            print(f" -> Cropped: {filename}")
            x, y, w, h = roi
            cropped_image = image[y:y+h, x:x+w]
        
        # If user pressed Enter/Space without drawing a box, roi[2] and roi[3] will be 0
        else:
            print(f" -> Accepted as-is: {filename}")
            cropped_image = image

        # Save the processed image (either cropped or as-is)
        dest_image_path = os.path.join(dest_dir, filename)
        os.makedirs(os.path.dirname(dest_image_path), exist_ok=True)
        cv2.imwrite(dest_image_path, cropped_image)
        new_data.append({'filename': filename, 'label': label})

    cv2.destroyAllWindows()

    if new_data:
        new_df = pd.DataFrame(new_data)
        new_csv_path = os.path.join(dest_dir, "_classes.csv")
        new_df.to_csv(new_csv_path, index=False)
        print(f"\nNew CSV saved for {os.path.basename(source_dir)} at: {new_csv_path}")

    return "CONTINUE"

if __name__ == "__main__":
    if not os.path.exists(DESTINATION_BASE_PATH):
        os.makedirs(DESTINATION_BASE_PATH)
        
    for data_set in SETS_TO_PROCESS:
        source_directory = os.path.join(SOURCE_BASE_PATH, data_set)
        destination_directory = os.path.join(DESTINATION_BASE_PATH, data_set)
        
        status = interactive_crop_and_save(source_directory, destination_directory)
        
        if status == "QUIT":
            print("\nUser quit the process. Halting script.")
            break

    print("\n--- Cropping process finished or was quit by user! ---")
    print(f"Your cleaned, cropped dataset is in '{DESTINATION_BASE_PATH}'")