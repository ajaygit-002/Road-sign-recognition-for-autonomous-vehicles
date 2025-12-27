# Traffic Sign Recognition for Autonomous Vehicles

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![Model](https://img.shields.io/badge/model-43--class-green.svg)](my_model.h5)

A compact, easy-to-use repository for recognizing traffic signs using a pretrained CNN and a PyQt5 GUI. This README is formatted to display well on all screen sizes — images and diagrams scale responsively.

<!-- toc -->
- [Quick Links](#quick-links)
- [Preview](#preview)
- [Requirements](#requirements)
- [Quickstart](#quickstart)
- [Training](#training)
- [Dataset Structure](#dataset-structure)
- [Project Layout](#project-layout)
- [Tips & Troubleshooting](#tips--troubleshooting)
- [License & Contact](#license--contact)
<!-- tocstop -->

## Quick Links
- Model: `my_model.h5`
- Training script: `training/train.py`
- Dataset folder: `dataset/`
- Visual summary: `VISUAL_SUMMARY.md`

## Preview
<div>
  <img src="VISUAL_SUMMARY.md" alt="visual summary" style="max-width:100%;height:auto;display:block;margin:0 auto;">
</div>

> Tip: images in this README use `max-width:100%` so they scale on mobile and desktop.

## Requirements
- Python 3.9 or newer
- Recommended packages (install with):

```bash
pip install -r requirements.txt
```

If you prefer a one-liner:

```bash
pip install tensorflow==2.13.0 keras numpy pillow scikit-learn matplotlib pyqt5
```

## Quickstart
1. Open a terminal in the project root.
2. Run the GUI:

```bash
python main.py
```

3. In the GUI: upload an image and get an immediate prediction.

Command-line training (optional):

```bash
python training/train.py
```

## Training
- Prepare dataset (see next section).
- Training saves a new model as `my_model_new.h5` and writes `Accuracy1.png` / `Loss1.png` to `outputs/`.
- Recommended callbacks: early stopping and LR reduction.

## Dataset Structure
Use the included `dataset/` layout or follow this minimal structure:

```
dataset/
  train/
    images/
    labels/  # YOLO-style .txt files or per-class folders depending on your pipeline
```

For classification-only training you can also organize images per-class under `dataset/train/<class_id>/`.

## Project Layout
- `main.py` — GUI + inference
- `training/train.py` — training entrypoint
- `utils/` — helper modules (`data_loader.py`, `model.py`, `classes.py`)
- `models/` — saved model artifacts (`traffic_sign_model.h5`, `my_model.h5`)
- `dataset/` — sample dataset folders

## Tips & Troubleshooting
- If you see TensorFlow DLL errors on Windows, reinstall a matching TF wheel:

```bash
pip uninstall tensorflow keras -y
pip install tensorflow==2.13.0
```

- If PyQt5 installation fails:

```bash
pip install PyQt5 --upgrade
```

- Use clear, front-facing sign images for best results. The model expects 30×30 RGB input — preprocessing is handled in `utils/`.

## License & Contact
This repository is provided as-is. See `LICENSE` for details.

Questions or improvements? Open an issue or contact the maintainer.
# Traffic Sign Recognition System

A PyQt5-based GUI application that recognizes traffic signs using a pre-trained deep learning model. The system can classify 43 different types of German traffic signs.

## Features

- **Pre-trained Model**: Includes a pre-trained CNN model (`my_model.h5`) for immediate traffic sign prediction
- **GUI Interface**: User-friendly PyQt5 interface for uploading and classifying images
- **Training Capability**: Can train new models with custom dataset (optional)
- **43 Traffic Sign Classes**: Supports recognition of 43 different German traffic sign types
- **Image Processing**: Automatic image resizing and preprocessing for accurate predictions

## Traffic Signs Recognized

The system recognizes traffic signs including:
- Speed limits (20km/h - 120km/h)
- No passing signs
- Priority roads
- Yield signs
- Stop signs
- No entry
- General caution
- Dangerous curves
- Road work
- Traffic signals
- Pedestrians crossing
- Bicycles crossing
- And many more (43 total classes)

## Installation

### Requirements
- Python 3.9+
- Windows OS (optimized for Windows)

### Setup

1. **Clone/Download** the project folder

2. **Install Python packages**:
```bash
pip install tensorflow==2.13.0 keras scikit-learn numpy matplotlib pillow pyqt5
```

Or install from requirements (if available):
```bash
pip install -r requirements.txt
```

## Run Commands

### Quick Start

To run the traffic sign recognition GUI application:

```bash
python main.py
```

This will launch the PyQt5 GUI interface where you can:
- Upload traffic sign images
- Get instant predictions
- Train new models (if dataset is available)

### Alternative Run Methods

**Run with Python directly**:
```bash
python main.py
```

**Run from PowerShell/Command Prompt**:
```powershell
cd "c:\Users\ajayo\OneDrive\Desktop\Road sign recognition for autonomous vehicles"
python main.py
```

**Train a New Model**:
```bash
python training/train.py
```

## Project Workflow

This project includes two main workflows:

### 1. **Inference Workflow (Image Classification)**

The main workflow for predicting traffic signs from images:

```
User Interface (PyQt5 GUI)
    ↓
1. Load Image
    ├─ Browse and select image file (PNG, JPG, JPEG, BMP)
    └─ Display image in GUI preview panel
    ↓
2. Preprocess Image
    ├─ Resize image to 30×30 pixels
    ├─ Convert to numpy array
    └─ Add batch dimension (1, 30, 30, 3)
    ↓
3. Load Pre-trained Model
    └─ Load 'my_model.h5' (43-class CNN model)
    ↓
4. Run Prediction
    ├─ Pass image through CNN
    ├─ Get probability scores for all 43 classes
    └─ Find class with highest probability
    ↓
5. Display Result
    └─ Show traffic sign name/class in GUI text area
```

**Key Steps in Code**:
- `loadImage()`: Handles file selection and image display
- `classifyFunction()`: Loads model, preprocesses image, runs prediction
- `get_sign_name()`: Maps class index to human-readable sign name

---

### 2. **Training Workflow (Model Training)**

Complete workflow for training a new model with custom dataset:

```
Dataset Preparation
    ├─ Required structure: dataset/train/
    │   ├─ images/ (contains image files)
    │   └─ labels/ (contains YOLO format .txt files)
    └─ Each label file contains: class_id x_center y_center width height
    ↓
1. Load Dataset
    ├─ Read all images from images/ directory
    ├─ Load corresponding label files from labels/ directory
    ├─ Resize all images to 30×30 pixels
    ├─ Extract class ID from YOLO format labels
    └─ Aggregate into data and labels arrays
    ↓
2. Data Preparation
    ├─ Convert data to numpy array
    ├─ Convert labels to numpy array
    ├─ Split data: 80% training, 20% testing
    ├─ Normalize/standardize images (optional)
    └─ Apply one-hot encoding to labels (43 classes)
    ↓
3. Model Creation
    ├─ Initialize Sequential CNN model
    ├─ Add 2 convolutional blocks (32 & 64 filters)
    ├─ Add max pooling and dropout layers
    ├─ Add fully connected dense layers
    └─ Configure output layer (43 units, softmax activation)
    ↓
4. Model Compilation
    ├─ Loss function: Categorical Crossentropy
    ├─ Optimizer: Adam
    └─ Metrics: Accuracy
    ↓
5. Model Training
    ├─ Batch size: 32
    ├─ Epochs: 15 (or until early stopping)
    ├─ Validation split: 20% of training data
    ├─ Callbacks:
    │   ├─ Early Stopping (monitor val_loss, patience=3)
    │   └─ Learning Rate Reduction (factor=0.5, patience=2)
    └─ Track accuracy and loss metrics
    ↓
6. Model Evaluation & Visualization
    ├─ Evaluate on test set
    ├─ Generate accuracy graph (training vs validation)
    ├─ Generate loss graph (training vs validation)
    └─ Save graphs as Accuracy1.png and Loss1.png
    ↓
7. Save Model
    └─ Save trained model as 'my_model_new.h5'
```

**Training Options**:

**Option A - GUI Training** (requires dataset to be loaded):
```bash
python main.py
# Click "Training" button in GUI
```

**Option B - Command Line Training** (recommended):
```bash
python training/train.py
```

**Key Functions**:
- `load_dataset()`: Loads images and labels from YOLO format
- `prepare_training_data()`: Splits and preprocesses data
- `create_model()`: Creates CNN architecture
- `compile_model()`: Configures loss and optimizer
- `model.fit()`: Trains the model with callbacks

---

### 3. **Data Flow Architecture**

```
Input Layer (30×30×3)
    ↓
Conv2D (32 filters, 5×5) + ReLU
    ↓
Conv2D (32 filters, 5×5) + ReLU + MaxPool2D + Dropout(0.25)
    ↓
Conv2D (64 filters, 3×3) + ReLU
    ↓
Conv2D (64 filters, 3×3) + ReLU + MaxPool2D + Dropout(0.25)
    ↓
Flatten
    ↓
Dense (256 units) + ReLU + Dropout(0.5)
    ↓
Dense (43 units) + Softmax
    ↓
Output (Probability distribution across 43 classes)
```

---

### 4. **File Processing Workflow**

| Stage | Input | Process | Output |
|-------|-------|---------|--------|
| **Load** | Image file (PNG/JPG) | Read from disk | Image data |
| **Resize** | Original image | Resize to 30×30 | Resized image |
| **Convert** | PIL Image | Convert to numpy | Numpy array |
| **Normalize** | Raw pixel values | Scale/normalize | Normalized array |
| **Predict** | Neural network | Forward pass | Class probabilities |
| **Decode** | Class index | Map to class name | Sign name |

---

## Usage

### Running the Application

1. **Open PowerShell/Command Prompt** in the project folder

2. **Run the program**:
```bash
python main.py
```

3. **Using the GUI**:
   - **Upload Image**: Click the upload button to select a traffic sign image
   - **Predict**: The system will automatically classify the sign
   - **View Result**: The traffic sign type will be displayed in the text area

### Training a New Model (Optional)

If you have a training dataset:

1. **Prepare Dataset Structure**:
```
dataset/
└── train/
    ├── 0/     (Speed limit 20km/h images)
    ├── 1/     (Speed limit 30km/h images)
    ├── 2/
    ├── 3/
    └── ... up to 42/
```

2. **Click "Train Model"** button in the GUI
   - The system will train a new CNN model
   - Saves trained model as `my_model_new.h5`
   - Generates accuracy and loss graphs

## Project Structure

```
New folder/
├── main.py              # Main application file with GUI and model logic
├── my_model.h5          # Pre-trained model (43-class traffic sign classifier)
├── README.md            # This file
└── dataset/ (optional)  # Training data folder
    └── train/
        ├── 0/
        ├── 1/
        └── ... 42/
```

## Model Architecture

The CNN model uses the following architecture:
- **Input**: 30×30 RGB images
- **Conv Layer 1**: 32 filters, 5×5 kernel + MaxPool + Dropout
- **Conv Layer 2**: 32 filters, 5×5 kernel + MaxPool + Dropout
- **Conv Layer 3**: 64 filters, 3×3 kernel + MaxPool + Dropout
- **Dense Layer 1**: 256 units with dropout
- **Output Layer**: 43 units (softmax activation)

## Troubleshooting

### TensorFlow DLL Error
If you encounter "DLL load failed" error:
```bash
pip uninstall tensorflow keras -y
pip install tensorflow==2.13.0
```

### Dataset Not Found
The program will skip training if the dataset folder is missing. You can still use the pre-trained model for predictions.

### PyQt5 Issues
If PyQt5 doesn't install properly:
```bash
pip install PyQt5 --upgrade
```

## File Outputs

When training a new model, the system generates:
- `my_model_new.h5` - Trained model weights
- `Accuracy1.png` - Training/validation accuracy graph
- `Loss1.png` - Training/validation loss graph

## Example Predictions

The system provides instant predictions for uploaded traffic sign images and displays the corresponding sign name (e.g., "Speed limit (50km/h)", "Stop", "Yield").

## Dependencies

| Package | Version |
|---------|---------|
| TensorFlow | 2.13.0 |
| Keras | 2.13.1 |
| NumPy | 1.24.3 |
| Pillow | 9.x+ |
| scikit-learn | Latest |
| PyQt5 | Latest |

## License

This project uses the German Traffic Sign Recognition Benchmark (GTSRB) dataset format.

## Notes

- Images are automatically resized to 30×30 pixels for model input
- The model expects standard RGB traffic sign images
- For best accuracy, use clear images of traffic signs
- Training a new model requires 30-60 minutes depending on dataset size

## Support

If you encounter issues:
1. Ensure Python 3.9+ is installed
2. Verify all dependencies are installed
3. Check that image paths are correct
4. Try reinstalling TensorFlow if DLL errors persist

---

**Created**: December 2025  
**Version**: 1.0
