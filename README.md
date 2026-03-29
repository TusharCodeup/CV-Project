# 🎨 AirCanvas Pro - 3D Gesture Drawing Studio with Structure from Motion

## 📌 Project Overview

AirCanvas Pro is an advanced computer vision project that integrates **Structure from Motion (SfM)** with **real-time hand gesture recognition** to create an interactive 3D drawing environment.

The system reconstructs **3D scenes from multiple 2D images** and allows users to **draw in air using hand gestures** captured through a webcam. It combines 3D reconstruction with natural human-computer interaction, eliminating the need for specialized hardware.

---

## ✨ Key Features

* 🎯 Gesture-Based Drawing – Draw using hand movements in real time
* 🏗️ 3D Reconstruction (SfM) – Convert multiple images into 3D point clouds
* 🖌️ Drawing Tools – Brush, Fill, Smudge, Text, Effects
* 🤝 Real-Time Interaction – Instant feedback using webcam input
* 📦 3D Integration – Use reconstructed 3D objects inside the drawing environment

---

## 🛠️ Tech Stack

* Python
* OpenCV (opencv-python & opencv-contrib-python)
* MediaPipe
* NumPy, SciPy
* Matplotlib

---

## ⚙️ System Requirements

* Python 3.8 or higher
* Webcam (required for gesture drawing)
* Minimum 8GB RAM
* OS: Windows / Linux / macOS

---

## 🚀 Installation Guide

### 1. Clone the Repository

```bash
git clone https://github.com/TusharCodeup/CV-Project.git
cd CV-Project
```

---

### 2. Create Virtual Environment (Recommended)

#### For Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

#### For Linux/Mac:

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run the Project

### Step 1: Generate Test Dataset

```bash
python data/download_dataset.py --type test
```

---

### Step 2: Run Structure from Motion (3D Reconstruction)

```bash
python main.py --image_dir data/test_sequence --output_dir outputs
```

---

### Step 3: Run AirCanvas Pro (Gesture Drawing)

```bash
python aircanvas_pro.py
```

👉 Make sure your webcam is enabled.

---

## 🎮 Gesture Controls

| Gesture           | Action       |
| ----------------- | ------------ |
| ☝️ Index Finger   | Draw         |
| ✌️ Index + Middle | Change Tool  |
| 👍 + Pinky        | Change Color |
| ✊ Fist            | Erase        |
| 🖐️ Open Hand     | Clear Canvas |

---

## 📂 Output Files

After running the SfM pipeline, the following files are generated in the `outputs/` folder:

* point_cloud.ply → 3D reconstructed model
* 3d_reconstruction.png → Visualization of 3D points
* camera_poses.json → Camera positions
* reconstruction_stats.json → Performance metrics

---

## 📁 Project Structure

```
CV-Project/
├── src/                  # SfM pipeline modules
├── data/                 # Dataset
├── outputs/              # Results
├── aircanvas_pro.py      # Gesture drawing app
├── main.py               # SfM pipeline
├── requirements.txt
└── README.md
```

---

## 🧠 Concepts Used

* Structure from Motion (SfM)
* SIFT Feature Detection
* Feature Matching (FLANN)
* RANSAC (Outlier Removal)
* Epipolar Geometry
* Triangulation (DLT)
* Bundle Adjustment
* Real-time Hand Tracking (MediaPipe)

---

## 📌 Notes

* Ensure good lighting for accurate gesture detection
* Keep camera stable during SfM image capture
* Use textured scenes for better 3D reconstruction

---

## 👨‍💻 Author

Tushar Codeup

---

## 📜 License

This project is for academic purposes.
