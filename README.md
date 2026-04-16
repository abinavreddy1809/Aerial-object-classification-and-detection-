# 🛸 Aerial Object Classification & Detection
### Bird vs Drone — Deep Learning Project | Labmentix Data Science Internship

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)
![Status](https://img.shields.io/badge/Status-Completed-green)

---

## 📌 Project Overview

This project develops a **deep learning-based solution** that classifies aerial images into two categories — **Bird 🦅** or **Drone 🚁** — and optionally performs **object detection** to locate these objects with bounding boxes in real-world scenes.

Accurate identification between birds and drones is critical for:
- 🛡️ **Security & Defense** — Detecting unauthorized drones in restricted airspace
- 🐦 **Wildlife Protection** — Monitoring birds near wind farms and airports
- ✈️ **Airport Safety** — Preventing bird-strike incidents on runways
- 🌿 **Environmental Research** — Tracking bird populations using aerial footage

---

## 🗂️ Repository Structure

```
aerial-classification/
│
├── 📓 aerial_classification_colab.py   # Full Google Colab training pipeline
├── 🌐 app.py                           # Streamlit deployment app
├── 📊 README.md                        # This file
│
├── models/                             # Saved model files (after training)
│   ├── best_aerial_model.keras         # Best classification model
│   └── runs/detect/bird_drone_yolo/
│       └── weights/best.pt             # YOLOv8 detection model
│
├── notebooks/
│   └── aerial_classification.ipynb     # Jupyter notebook version
│
├── outputs/                            # Generated charts and evaluation plots
│   ├── sample_images.png
│   ├── class_distribution.png
│   ├── augmented_samples.png
│   ├── eval_Custom_CNN.png
│   ├── eval_ResNet50_TL.png
│   └── model_comparison.png
│
└── requirements.txt                    # Python dependencies
```

---

## 📦 Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/aerial-classification.git
cd aerial-classification
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

**requirements.txt contents:**
```
tensorflow>=2.12.0
streamlit>=1.28.0
pillow>=9.0.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
pandas>=1.5.0
ultralytics>=8.0.0
opencv-python>=4.7.0
```

### 3. Dataset Setup
Place your dataset in the following structure:
```
classification_dataset/
├── TRAIN/
│   ├── bird/        # 1,414 images
│   └── drone/       # 1,248 images
├── VALID/
│   ├── bird/        # 217 images
│   └── drone/       # 225 images
└── TEST/
    ├── bird/        # 121 images
    └── drone/       # 94 images

object_detection_Dataset/
├── train/
│   ├── images/      # 2,662 images
│   └── labels/      # YOLOv8 .txt annotations
├── val/
│   ├── images/      # 442 images
│   └── labels/
└── test/
    ├── images/      # 215 images
    └── labels/
```

---

## 🚀 Usage

### Option A — Run Training in Google Colab (Recommended)

1. Upload `aerial_classification_colab.py` to Google Colab
2. Mount your Google Drive with the dataset
3. Update the `DATASET_PATH` variable at the top of the script
4. Enable GPU: **Runtime → Change runtime type → GPU**
5. Run all cells sequentially

```python
# Update this path in the script to match your Drive location
DATASET_PATH = "/content/drive/MyDrive/labmentix_project/classification_dataset"
```

### Option B — Run the Streamlit App (After Training)

```bash
# 1. Make sure best_aerial_model.keras is in the same folder as app.py
# 2. Launch the app
streamlit run app.py

# 3. Open in browser
# http://localhost:8501
```

---

## 🧠 Models Built

### Model A — Custom CNN (Built from Scratch)

A custom Convolutional Neural Network designed to learn features directly from aerial images.

| Component | Details |
|-----------|---------|
| Architecture | 3 × Conv Blocks (32→64→128 filters) + Dense layers |
| Regularization | BatchNormalization + Dropout (0.25–0.5) |
| Optimizer | Adam (lr=0.001) |
| Loss | Binary Crossentropy |
| Output | Sigmoid (binary: Bird/Drone) |

### Model B — ResNet50 Transfer Learning (Best Model ⭐)

Leverages ResNet50 pretrained on ImageNet, fine-tuned on our aerial dataset.

| Component | Details |
|-----------|---------|
| Base | ResNet50 (pretrained on ImageNet, 1.28M images) |
| Phase 1 | Frozen base — train top classifier layers only |
| Phase 2 | Unfreeze last 30 ResNet50 layers for fine-tuning |
| Optimizer | Adam (Phase 1: lr=5e-4, Phase 2: lr=1e-5) |

### Model C — YOLOv8 Object Detection (Optional)

Detects and localizes birds/drones with bounding boxes in real-world scenes.

| Component | Details |
|-----------|---------|
| Base | YOLOv8n (nano — fastest variant) |
| Input size | 640 × 640 |
| Output | Bounding box + class + confidence |
| Metric | mAP50, mAP50-95 |

---

## 🔄 Project Workflow

```
Raw Dataset
    │
    ▼
1. EXPLORE          → Count images, check imbalance, visualize samples
    │
    ▼
2. PREPROCESS       → Resize to 224×224, normalize pixels to [0, 1]
    │
    ▼
3. AUGMENT          → Rotation, flip, zoom, brightness (training set only)
    │
    ▼
4. BUILD MODELS     → Custom CNN + ResNet50 Transfer Learning
    │
    ▼
5. TRAIN            → EarlyStopping + ModelCheckpoint + ReduceLROnPlateau
    │
    ▼
6. EVALUATE         → Confusion Matrix, F1-Score, Accuracy/Loss curves
    │
    ▼
7. COMPARE          → Side-by-side accuracy, F1, training time
    │
    ▼
8. DEPLOY           → Streamlit app with image upload + predictions
    │
    ▼ (Optional)
9. DETECT           → YOLOv8 bounding box detection on real scenes
```

---

## 📊 Results Summary

> Note: Actual metrics depend on your training run, GPU, and random seed.

| Model | Test Accuracy | F1-Score | Training Time | Model Size |
|-------|--------------|---------|--------------|-----------|
| Custom CNN | ~87–92% | ~0.87–0.92 | ~15 min | ~10 MB |
| **ResNet50 TL** | **~93–97%** | **~0.93–0.97** | ~30–40 min | ~90 MB |
| YOLOv8 (mAP50) | ~0.80–0.90 | — | ~20 min | ~6 MB |

**🏆 Best Model: ResNet50 Transfer Learning** — highest accuracy and F1-score, lowest overfitting risk.

---

## 🌐 Streamlit App Features

| Tab | Features |
|-----|---------|
| 🔍 Single Image | Upload image → Bird/Drone prediction → Confidence score → Probability bar chart → Optional YOLO bounding boxes |
| 📂 Batch Mode | Upload multiple images → Results table → Summary stats → Pie chart → CSV download |
| 📊 Project Info | Dataset stats, model info, workflow summary, tech stack |

---

## 📈 Training Callbacks Used

| Callback | Purpose |
|----------|---------|
| `EarlyStopping(patience=7)` | Stops training when val_accuracy stops improving |
| `ModelCheckpoint` | Saves only the best model (by val_accuracy) |
| `ReduceLROnPlateau(patience=4)` | Halves learning rate when val_loss plateaus |

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| Language | Python 3.9+ |
| Deep Learning | TensorFlow / Keras |
| Object Detection | YOLOv8 (Ultralytics) |
| Data Processing | NumPy, Pandas, scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Deployment | Streamlit |
| Training Environment | Google Colab (GPU) |

---

## 🔮 Future Improvements

- [ ] Try **EfficientNetB0** or **MobileNetV2** as lightweight alternatives
- [ ] Add **Grad-CAM** heatmaps to visualize model attention areas
- [ ] Expand dataset with edge cases (drones at high altitude, birds in flight)
- [ ] Deploy YOLOv8 on **live webcam or drone feed** for real-time monitoring
- [ ] Add **multi-class detection** (bird species, drone model type)
- [ ] Host on **Streamlit Cloud** or **Hugging Face Spaces** for public access

---

## 📁 Key Files Description

| File | Description |
|------|-------------|
| `aerial_classification_colab.py` | Complete 7-part training pipeline — exploration, preprocessing, augmentation, model building, training, evaluation, comparison, and YOLOv8 setup |
| `app.py` | Streamlit web application with 3-tab dark-themed UI for classification and detection |
| `best_aerial_model.keras` | Saved best-performing model (generated after training — not included in repo due to size) |

---

## 👤 Author

**[Your Name]**
Data Science Intern @ **Labmentix**
April 2026

---

## 📄 License

This project was developed as part of the Labmentix Data Science Internship program. All code is for educational purposes.

---

*Built with ❤️ and a lot of Googling during my first deep learning internship* 😄
