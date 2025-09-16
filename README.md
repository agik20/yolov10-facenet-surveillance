# Criminal Suspect Localization Using CCTV-Based Face Recognition and Object Detection

![YOLOv10](https://img.shields.io/badge/YOLOv10-Face%20Detection-00FFFF?style=for-the-badge)
![FaceNet](https://img.shields.io/badge/FaceNet-Face%20Recognition-FF00FF?style=for-the-badge)
![SVM](https://img.shields.io/badge/SVM-Classification-FFFF00?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/OpenCV-Real--time-00FF00?style=for-the-badge)

**Team Name:** Gotham  
**Members:**
- Ardutra Agi Ginting (ardutraa40@gmail.com)
- Muhammad Abyan Nurfajarizqi (muhammadabyan077@gmail.com)
- Muhammad Hafidz Hidayatullah (hafidzhidayatullah1012@gmail.com)

**Origin:** Islamic University of Indonesia

---

## 📌 Project Overview

**"Identifying Suspect Locations Automatically Using Face Detection and Face Recognition"**

This project provides significant benefits in supporting law enforcement through the use of Face Detection and Face Recognition technology on CCTV to track the location of suspects more quickly and accurately. The system automates the monitoring process that was previously performed manually, reducing officer workload and improving tracking efficiency.

> ⚠️ **Important Note on NIK Usage:**  
> The NIK integration described here is a **conceptual design only**. This project does **not** connect to real government databases. Instead, a **mock database** of individual reference images is used to simulate the process. Actual national ID data remains under the authority of government institutions.

---

## 📖 Abstract

This project presents an AI-based surveillance system for automatically locating suspects through CCTV. The system applies YOLOv10s for real-time face detection and FaceNet with SVM for accurate face recognition using embeddings comparison. By integrating suspect data via National Identification Numbers (NIK), the system automates monitoring, reduces officer workload, and enhances tracking efficiency.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    A[Input Suspect NIK] --> B[Searching NIK in Database]
    B --> C[Retrieve Reference Face by NIK]
    D[Streaming CCTV] --> E[Face Detection]
    E --> F{Face Recognition<br>Does Detected Face Match Reference Face?}
    C --> F
    F --|Match| G[Store Matched Face + Location Info]
    F --|Not Match| E
    G --> H[Display Matched Face and Location on Website]
    H --> I[Finish]
```

## ⚙️ Technical Implementation

### 1. Face Detection – **YOLOv10s**
- **Purpose:** Real-time face detection in CCTV feeds
- **Advantages:** Lightweight, optimized for real-time inference
- **Input:** CCTV video stream frame-by-frame
- **Output:** Cropped face regions with bounding box coordinates

### 2. Face Recognition – **FaceNet**
- **Purpose:** Generate 128-dimensional embedding vectors for each face
- **Advantages:** State-of-the-art identity representation
- **Process:** Converts detected faces into comparable embeddings

### 3. Classification – **Support Vector Machine (SVM)**
- **Purpose:** Match detection using embedding comparisons
- **Advantages:** Effective with high-dimensional data
- **Output:** Match/No Match decision with confidence score

### 4. Database Integration
- **NIK-based Search:** Reference face retrieval using mock NIK database
- **Metadata Storage:** Timestamp, CCTV location, and detection evidence

---

## 📈 Performance Evaluation

### Confusion Matrix (CPU Execution)
![Confusion Matrix](https://github.com/user-attachments/assets/46be7671-8ed9-427f-b726-b354085f4026)

*The confusion matrix demonstrates perfect classification performance with no misclassifications*

### Metrics Report (GPU Execution)
```json
{
  "accuracy": 1.0,
  "macro_avg": {
    "precision": 1.0,
    "recall": 1.0,
    "f1": 1.0
  },
  "weighted_avg": {
    "precision": 1.0,
    "recall": 1.0,
    "f1": 1.0
  },
  "per_class": {
    "abyan": { "precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 252 },
    "agi": { "precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 253 },
    "apis": { "precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 252 }
  }
}
```

---

## 🖥️ Web Application Features

| Feature | Description |
|---------|-------------|
| **Home Page** | System overview and capabilities |
| **CCTV Page** | Real-time streaming from multiple cameras with location selection |
| **Search Page** | NIK-based search for suspect identification |
| **History Page** | Complete detection records with filtering options |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- CUDA-enabled GPU (recommended)
- Web camera or CCTV feed access

### Installation
```bash
# Clone the repository
git clone https://github.com/agik20/criminal-suspect-localization.git
cd criminal-suspect-localization

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/MacOS
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Usage
```bash
# Run the web application
python app.py

# Start face detection and recognition
python detect.py --source 0  # webcam
# or
python detect.py --source rtsp://camera_feed_url  # CCTV stream
```

---

## 📊 Results Interpretation

The system achieves perfect classification performance across all evaluation metrics:
- **100% accuracy** in face recognition tasks
- **Perfect precision and recall** for all tested classes
- **Robust performance** on both CPU and GPU environments

---

## 🔮 Future Enhancements

- Integration with larger CCTV networks
- Advanced tracking algorithms across multiple cameras
- Enhanced database synchronization capabilities
- Mobile application for field officers

---

## 📝 License

This project is developed for academic purposes. Commercial use requires proper authorization and compliance with privacy regulations.

---

## 🤝 Contributing

We welcome contributions to enhance this project:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Submit a pull request

---

<div align="center">

**⭐ Star this repository if you find it useful!**

*For questions and support, please open an issue in the GitHub repository*

</div>
