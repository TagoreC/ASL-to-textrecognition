# ASL to Text Recognition System

> **Real-time American Sign Language (ASL) recognition using Convolutional Neural Networks (CNN) and OpenCV**
---

## 📌 Table of Contents

- [About the Project](#about-the-project)
- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Future Enhancements](#future-enhancements)
  

---

## 📖 About the Project

The **ASL to Text Recognition System** is a computer vision project that bridges the communication gap between the hearing-impaired community and others. Using a webcam, the system captures hand gestures corresponding to American Sign Language (ASL) alphabets (A–Z) and converts them into readable text in real time.

This project addresses a critical accessibility need — enabling people with hearing or speech disabilities to communicate more effectively through sign language, which is then translated into text using deep learning and image processing techniques.

> The system recognizes **27 classes** — the blank/neutral gesture (`0`) and all **26 letters of the ASL alphabet (A–Z)**.

---

## ✨ Features

- 🎥 **Real-time recognition** via webcam using OpenCV
- 🧠 **CNN-based deep learning model** trained on ASL hand gesture images
- 🔤 **A–Z alphabet detection** with a blank/neutral class
- 📦 **Prediction smoothing** using a rolling buffer of the last 15 frames
- 🖊️ **Automatic sentence building** — letters are auto-captured after stable detection
- 📊 **Confidence score display** for each prediction
- 🖥️ **Tkinter GUI** for a user-friendly interface
- ⚡ **Preprocessing pipeline** with Gaussian blur, adaptive thresholding, and normalization

---

## 🏗️ Architecture

### System Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    INPUT LAYER                          │
│           Webcam Feed (Real-time Video)                 │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│                  PREPROCESSING MODULE                   │
│  1. ROI Extraction (100,100) → (400,400)                │
│  2. Grayscale Conversion (BGR → Gray)                   │
│  3. Gaussian Blur  (5×5 kernel, σ=2)                    │
│  4. Adaptive Thresholding (Gaussian, THRESH_BINARY_INV) │
│  5. Resize to 128×128 pixels                            │
│  6. Normalization (pixel values ÷ 255.0)               │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│              CNN MODEL (sign_model.keras)                │
│                                                         │
│  Input: (128, 128, 1)                                   │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Conv2D(32, 3×3, ReLU)                          │   │
│  │  MaxPooling2D(2×2)                              │   │
│  └─────────────────────────────────────────────────┘   │
│                          ↓                              │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Conv2D(64, 3×3, ReLU)                          │   │
│  │  MaxPooling2D(2×2)                              │   │
│  └─────────────────────────────────────────────────┘   │
│                          ↓                              │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Conv2D(128, 3×3, ReLU)                         │   │
│  │  MaxPooling2D(2×2)                              │   │
│  └─────────────────────────────────────────────────┘   │
│                          ↓                              │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Flatten()                                      │   │
│  │  Dense(128, ReLU)                               │   │
│  │  Dropout(0.5)                                   │   │
│  │  Dense(27, Softmax)  ← Output: 27 classes       │   │
│  └─────────────────────────────────────────────────┘   │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│              PREDICTION & SMOOTHING                     │
│  • Rolling Buffer (maxlen=15 frames)                    │
│  • Majority Voting over buffer                          │
│  • Stable prediction after 20+ consistent frames       │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│                   OUTPUT LAYER                          │
│  • Recognized Letter displayed on frame                 │
│  • Confidence % shown on screen                         │
│  • Sentence built character by character                │
│  • Tkinter UI or OpenCV window display                  │
└─────────────────────────────────────────────────────────┘
```

### CNN Model Architecture Details

| Layer | Type | Filters/Units | Kernel | Activation | Output Shape |
|-------|------|---------------|--------|------------|--------------|
| Input | — | — | — | — | (128, 128, 1) |
| Conv2D_1 | Convolution | 32 | 3×3 | ReLU | (126, 126, 32) |
| MaxPool_1 | Pooling | — | 2×2 | — | (63, 63, 32) |
| Conv2D_2 | Convolution | 64 | 3×3 | ReLU | (61, 61, 64) |
| MaxPool_2 | Pooling | — | 2×2 | — | (30, 30, 64) |
| Conv2D_3 | Convolution | 128 | 3×3 | ReLU | (28, 28, 128) |
| MaxPool_3 | Pooling | — | 2×2 | — | (14, 14, 128) |
| Flatten | — | — | — | — | (25088,) |
| Dense_1 | Fully Connected | 128 | — | ReLU | (128,) |
| Dropout | Regularization | — | — | — | (128,) — 50% drop rate |
| Dense_2 (Output) | Fully Connected | 27 | — | Softmax | (27,) |

- **Optimizer:** Adam  
- **Loss Function:** Categorical Cross-Entropy  
- **Metrics:** Accuracy  
- **Epochs:** 10  
- **Validation Split:** 20%

---

## 📁 Project Structure

```
ASL-to-Text-Recognition/
│
├── CV.ipynb                   # Jupyter Notebook — data loading, preprocessing & model training
├── realtime_prediction.py     # Real-time webcam-based ASL detection with sentence building
├── test.py                    # Manual key-press based testing script
├── ui.py                      # Tkinter GUI launcher
├── fun.py                     # MediaPipe API test/utility script
├── sign_model.keras            # Trained CNN model (saved in Keras format)
│
├── dataSet/                   # Dataset directory (not included in repo — see Dataset section)
│   ├── trainingData/
│   │   ├── 0/                 # Blank/neutral gesture
│   │   ├── A/
│   │   ├── B/
│   │   └── ... Z/
│   └── testingData/
│       ├── 0/
│       ├── A/
│       └── ... Z/
│
└── README.md
```

---

## 📊 Dataset

The dataset consists of grayscale hand gesture images representing the ASL alphabet.

| Split | Samples |
|-------|---------|
| Training | ~12,853 images |
| Testing | ~4,268 images |

- **Image Size:** 128×128 pixels (grayscale)
- **Classes:** 27 (blank `0` + A–Z)
- **Preprocessing applied during loading:**
  - BGR → Grayscale
  - Gaussian Blur (5×5, σ=2)
  - Adaptive Gaussian Thresholding (binary inverse)
  - Resize to 128×128
  - Pixel normalization [0, 1]

> **Note:** The dataset is not included in this repository due to size constraints. You can collect your own using standard ASL gesture datasets available on Kaggle.

---

## 🏋️ Model Training

The model is trained inside `CV.ipynb` (Google Colab with T4 GPU support):

```python
# Data Loading
X_train, y_train = load_dataset('trainingData')  # Shape: (12853, 128, 128, 1)
X_test, y_test   = load_dataset('testingData')   # Shape: (4268, 128, 128, 1)

# Model Training
model.fit(X, y, epochs=10, validation_split=0.2)

# Model Saving
model.save("sign_model.keras")
```

### Training Performance

| Epoch | Train Accuracy |
|-------|---------------|
| 1 | 32.85% |
| 5 | 96.82% |
| 10 | **98.60%** |

> The model achieves **~98.6% training accuracy**. High train-test divergence may indicate the model trained on a specific data distribution; further data augmentation and regularization can improve generalization.

---

## 🚀 Installation

### Prerequisites

- Python 3.8+
- Webcam / Camera

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/ASL-to-Text-Recognition.git
cd ASL-to-Text-Recognition
```

### 2. Install Dependencies

```bash
pip install tensorflow opencv-python numpy mediapipe
```

Or install from a requirements file:

```bash
pip install -r requirements.txt
```

> **`requirements.txt`** (create this manually if needed):
> ```
> tensorflow>=2.10
> opencv-python>=4.5
> numpy>=1.21
> mediapipe>=0.9
> ```

### 3. Ensure the Model File is Present

Make sure `sign_model.keras` is in the root directory of the project.

---

## 🖥️ Usage

### ▶️ Real-time Prediction (Recommended)

```bash
python realtime_prediction.py
```

- Place your hand inside the **green ROI box** on screen.
- The system will **automatically detect and capture letters** after 20 stable frames.
- **Press `Q`** to quit and see the final sentence.

### 🧪 Manual Test Mode

```bash
python test.py
```

| Key | Action |
|-----|--------|
| `C` | Capture current letter into sentence |
| `S` | Add a space |
| `D` | Delete last character |
| `Q` | Quit |

### 🖼️ GUI Launcher

```bash
python ui.py
```

Launches a Tkinter window with a **"Start Camera"** button.

---

## 📈 Results

- **Model Training Accuracy:** ~98.6%
- **Real-time detection** with prediction smoothing via 15-frame rolling buffer
- **Confidence scores** displayed live during recognition
- Automatic sentence construction character-by-character

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| **Python ** | Core programming language |
| **TensorFlow / Keras** | Deep learning model (CNN) training and inference |
| **OpenCV** | Webcam capture, image processing, display |
| **NumPy** | Numerical operations and array manipulation |
| **MediaPipe** | Utility/exploratory hand landmark detection |
| **Tkinter** | GUI interface for the application |
| **Google Colab** | Cloud-based model training with GPU (T4) |

---

## 🔮 Future Enhancements

- [ ] Integrate **MediaPipe Hand Landmarks** for more robust hand detection
- [ ] Support **dynamic gestures** (not just static letters)
- [ ] Expand to **full ASL words and phrases**, not just alphabets
- [ ] Add **text-to-speech** output for accessibility
- [ ] Deploy as a **web application** using Flask/FastAPI + WebRTC
- [ ] Improve generalization with **data augmentation** and **transfer learning**
- [ ] Support **multiple hand detection** (two-handed signs)

---

##  Acknowledgements

- ASL dataset contributors and the open-source community
- TensorFlow and Keras teams for their amazing deep learning frameworks
- OpenCV for real-time computer vision capabilities
- Google Colab for providing free GPU compute for training


