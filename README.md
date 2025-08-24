# Criminal Suspect Localization Using CCTV-Based Face Recognition and Object Detection

**Team Name:** Gotham  
**Members:**  
- Ardutra Agi Ginting (ardutraa40@gmail.com)  
- Muhammad Abyan Nurfajarizqi (muhammadabyan077@gmail.com)  
- Muhammad Hafidz Hidayatullah (hafidzhidayatullah1012@gmail.com)  

**Origin:** Islamic University of Indonesia

---

## 📌 Project Description
“Identifying Suspect Locations Automatically Using Face Detection and Face Recognition”

This project provides significant benefits in supporting law enforcement through the use of Face Detection and Face Recognition technology on CCTV to track the location of suspects more quickly and accurately. In addition, the system is capable of automating the monitoring process that was previously performed manually, thereby reducing the workload of officers and improving tracking efficiency. From a sustainability perspective, the system has the potential to be further expanded by integrating with a larger network of CCTV cameras across various regions in Indonesia and synchronizing with the national database.

---

## 📖 Abstract
This project presents an AI-based surveillance system for automatically locating suspects through CCTV. The system applies YOLOv10s for real-time face detection and FaceNet with SVM for accurate face recognition using embeddings comparison. By integrating suspect data via National Identification Numbers (NIK), the system automates monitoring, reduces officer workload, and enhances tracking efficiency. Designed for scalability, it offers a practical and innovative solution to strengthen law enforcement and public safety.

---

## ⚙️ System Workflow
<img width="810" height="609" alt="gotham drawio" src="https://github.com/user-attachments/assets/3dd02f96-f20e-4e9c-9ecb-8ce60053bd90" />

1. **Input:** User provides suspect’s **NIK**.  
2. **Database Retrieval:** System retrieves reference face image from database.  
3. **CCTV Analysis:**  
   - YOLOv10s detects faces from each CCTV frame.  
   - Detected faces → processed by FaceNet to generate embeddings.  
   - SVM classifier compares embeddings against the reference.  
4. **Decision:**  
   - If similarity score > threshold → Match confirmed.  
   - Metadata (timestamp, CCTV location) stored in database.  
5. **Output:**  
   - Cropped face + location displayed on website.  
   - History of detections logged for future queries.  

---

## 🖥️ Web Application Features
- **Home Page** → Overview of the system.  
- **CCTV Page** → Real-time streaming from multiple CCTV cameras with location selection.  
- **Search Page** → Search by NIK to quickly identify suspect appearances.  
- **History Page** → Complete record of past detections with filtering options.  

---

## ⚙️ Methodology & Models

### 1. Face Detection – **YOLOv10s**
- **Why YOLOv10s?**
  - Lightweight, optimized for real-time inference.
  - Balances accuracy and speed, ideal for CCTV video feeds.
- **Role in Pipeline:** Detects bounding boxes for all visible faces in each CCTV frame.
- **Input:** CCTV video stream frame-by-frame.  
- **Output:** Cropped face regions with bounding box coordinates.  

---

### 2. Face Recognition – **FaceNet**
- **Why FaceNet?**
  - Produces a 128-dimensional embedding vector for each face.
  - State-of-the-art in robust identity representation.  
  - Embeddings allow **cosine similarity / Euclidean distance** comparison.  
- **Process:**
  - Each detected face → converted into embeddings via FaceNet.
  - Embeddings stored in database for known suspects.  

---

### 3. Classification – **Support Vector Machine (SVM)**
- **Why SVM?**
  - Effective for classification tasks with high-dimensional embeddings.
  - Performs well with limited labeled training data (suspect face images).
- **Process:**
  - Embeddings → classified against reference embeddings.
  - Output: **Match / No Match** decision.  

---

### 4. Database Integration
- **NIK-based Search:**  
  User provides NIK → reference face image retrieved from database → embeddings generated → stored for comparison.  
- **Metadata Storage:**  
  Detection timestamp, CCTV ID, location, and cropped face stored for traceability. 

## 📈 Evaluation Results

The system was evaluated in two settings: **CPU (confusion matrix visualization)** and **GPU (JSON-based metrics report)**.

---

### 1. Confusion Matrix (CPU)
The confusion matrix below illustrates the classification results of the **Face Recognition model** when executed on CPU.  

Each row represents the **true class**, and each column represents the **predicted class**.  
A perfect diagonal line indicates that the classifier correctly recognized all faces without misclassifications.

<img width="1600" height="900" alt="evaluation_report" src="https://github.com/user-attachments/assets/46be7671-8ed9-427f-b726-b354085f4026" />

*Result:* The confusion matrix shows perfect classification performance, with no off-diagonal misclassifications.

---

### 2. Metrics Report (GPU)
On GPU execution, the system was able to achieve **perfect performance** with the following metrics:

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
    "abyan": {
      "precision": 1.0,
      "recall": 1.0,
      "f1": 1.0,
      "support": 252
    },
    "agi": {
      "precision": 1.0,
      "recall": 1.0,
      "f1": 1.0,
      "support": 253
    },
    "apis": {
      "precision": 1.0,
      "recall": 1.0,
      "f1": 1.0,
      "support": 252
    }
  }
}
