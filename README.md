# Multimodal Counterfeit Medicine Project

## Project Overview
This project provides a comprehensive solution for detecting counterfeit medicines using a multimodal approach. It combines computer vision (Deep Learning) with Optical Character Recognition (OCR) to analyze medicine packaging and verify authenticity. The system uses models like EfficientNet and custom CNNs, integrated with explainability tools (XAI) to provide transparent results.

## Key Features
- **Multimodal Analysis:** Integrates visual features from medicine packaging with textual data extracted via OCR.
- **Deep Learning Models:** Utilizes state-of-the-art architectures including EfficientNet and custom TensorFlow/Keras models.
- **OCR Integration:** Supports multiple OCR engines including Tesseract and PaddleOCR for robust text extraction in various conditions.
- **Explainable AI (XAI):** Uses `tf-explain` to visualize model focus and improve interpretability.
- **Web Interface:** Includes a Flask-based web application for easy interaction and real-time medicine verification.

## Technologies Used
- **Deep Learning:** TensorFlow, Keras, PyTorch
- **Computer Vision:** OpenCV, PaddleOCR, EasyOCR, Tesseract
- **Data Science:** Pandas, NumPy, Scikit-learn
- **Visualization:** Matplotlib, Seaborn
- **Web Framework:** Flask

## Project Structure
```text
Multimodal Counterfeit Medicine Project/
├── PROJECT (1).ipynb          # Main research and experimentation notebook
├── app_final_optimized_finetuned.py  # Final optimized Flask web application
├── train_final_paddle_robust_cpu.py  # Final robust training script (PaddleOCR)
├── model/                      # Folder for weights (Ignored in Git, see Releases)
├── results/                    # Evaluation plots (Confusion Matrix, ROC, etc.)
├── samples/                    # Sample images for testing
├── archive/                    # Experimental and legacy scripts
├── templates/                  # HTML templates for the web app
├── uploads/                    # Temporary storage for uploaded images
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```

## Getting Started

### Prerequisites
- Python 3.10+
- Tesseract OCR (if using Tesseract-based scripts)

### Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage
- **Training:** Run `python train_final_paddle_robust_cpu.py` or the Jupyter Notebook to train/fine-tune models.
- **Deployment:** Start the web app using `python app_final_optimized_finetuned.py` and navigate to the local server.
