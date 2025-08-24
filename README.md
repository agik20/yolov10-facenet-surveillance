# yolov10-facenet-surveillance

# CCTV-Based Suspect Tracking System

**Team Name:** Gotham  
**Members:**  
- Ardutra Agi Ginting (ardutraa40@gmail.com)  
- Muhammad Abyan Nurfajarizqi (muhammadabyan077@gmail.com)  
- Muhammad Hafidz Hidayatullah (hafidzhidayatullah1012@gmail.com)  

**Origin:** Universitas Islam Indonesia (UII)  
**Repository Link:** gotham  

---

## 📌 Project Description
“Identifying Suspect Locations Automatically Using Face Detection and Face Recognition”

This project presents an AI-based surveillance system for automatically locating suspects through CCTV.  
The system applies **YOLOv10s** for real-time face detection and **FaceNet + SVM** for accurate face recognition using facial embeddings.  
By integrating suspect data via **National Identification Numbers (NIK)**, the system automates monitoring, reduces officer workload, and enhances tracking efficiency.  
Designed for scalability, it offers a practical and innovative solution to strengthen law enforcement and public safety.

---

## 📖 Abstract
The system leverages modern computer vision and deep learning models to process CCTV footage in real time:  
- **YOLOv10s** → Detects faces from CCTV video streams.  
- **FaceNet + SVM** → Recognizes suspects by comparing embeddings to reference data.  
- **Database Integration** → Matches recognized faces with NIK-based records.  
- **Web Dashboard** → Displays detections, location, and history.  

The result is a system capable of **automated suspect identification and localization** that supports law enforcement operations.

---

## ⚙️ System Workflow
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

## 📊 Project Architecture
```text
Input CCTV → YOLOv10s (Face Detection) → FaceNet (Embeddings) → SVM (Classification) 
   → Match with Database (NIK) → Website Dashboard (Result + History)
