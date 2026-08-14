# 🛡️ Smart Retail: Shoplifting Detection System

Welcome to the **Shoplifting Detection System**, a robust, end-to-end deep learning pipeline designed to automatically detect shoplifting behavior from CCTV footage. 

Built as a final-year project, this system leverages a hybrid **CNN + LSTM** architecture to extract spatial features from video frames and analyze their temporal sequence to identify suspicious activity.

---

## 🏗️ Architecture

This project is built using a modern, modular architecture:

1. **Data Preprocessing Pipeline** (`src/data_prep/`):
   - Extracts raw `.mp4` videos using OpenCV.
   - Resizes frames to `224x224`, converts them to RGB, and normalizes them for PyTorch.
   - Randomly splits the dataset into Training, Validation, and Testing sets using an 70/20/10 split.
   - Supports the **DCSASS Dataset** by reading CSV labels to automatically categorize videos into `normal` and `shoplifting`.

2. **Deep Learning Model** (`src/models/`):
   - **CNN Backbone**: A pre-trained `ResNet50` acts as a spatial feature extractor, processing each frame individually to understand *what* is happening in the scene.
   - **LSTM Head**: A Long Short-Term Memory network takes the sequence of features from the CNN and analyzes the temporal relationship to understand the *motion* and *behavior* over time.

3. **Training & Evaluation** (`src/training/`):
   - Supports GPU acceleration via CUDA.
   - Uses CrossEntropyLoss and an Adam optimizer.
   - Features **Early Stopping** to prevent overfitting.
   - Outputs robust classification reports and confusion matrices using `scikit-learn` and `seaborn`.

4. **Inference & UI Demo** (`src/inference/` & `src/demo/`):
   - Provides a standalone inference script (`predict.py`) to run predictions on raw videos.
   - Includes a beautiful **Streamlit Web Application** for a real-time, interactive demonstration.

---

## 🚀 Getting Started

### 1. Installation

Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Dataset Preparation
Ensure your raw videos are placed in `data/raw/shoplifting` and `data/raw/normal`. Then run:
```bash
python src/data_prep/extract_frames.py
python src/data_prep/make_splits.py
```

### 3. Training
To train the model (we recommend using Google Colab for GPU support):
```bash
python src/training/train.py
```

### 4. Evaluation
To evaluate your trained model on the unseen test set and generate a Confusion Matrix:
```bash
python src/training/evaluate.py
```

### 5. Streamlit Demo App
To launch the graphical user interface where you can upload and test your own videos:
```bash
streamlit run src/demo/app.py
```

---

## 📊 Limitations & Future Work

### Class Imbalance
The current iteration of the dataset (DCSASS) possesses a severe **Class Imbalance** (e.g., 741 Normal videos vs 155 Shoplifting videos). As a result, the model may struggle to generalize perfectly and can exhibit biases toward predicting one class over the other.

**Solutions implemented/proposed:**
- Use of PyTorch's `WeightedRandomSampler` or passing class weights to `CrossEntropyLoss` to heavily penalize misclassifications of the minority class.
- Data Augmentation (e.g., flipping, rotating, and cropping shoplifting videos) to artificially expand the minority class.
- Implementing Focal Loss instead of standard CrossEntropy.

Despite the data constraints, the **end-to-end pipeline** (from raw MP4 extraction to training, evaluation, and GUI deployment) is 100% functional and production-ready.

---
*Developed as a Final Year Academic Project.*
