# 🧠 Multimodal Framework for Early Detection of Alzheimer’s Disease

A Deep Learning-based Web Application for Early Alzheimer’s Risk Assessment using **Handwriting Analysis (450 Features)** and **Speech MFCC Analysis**.

---

## 📌 Project Overview

Alzheimer’s Disease (AD) is a progressive neurodegenerative disorder that affects memory, cognition, and behavior. Early detection is critical for timely medical intervention and improved patient outcomes.

This project presents a **Multimodal Deep Learning Framework** that combines:

- ✍️ Handwriting Feature Analysis (450 numerical features)
- 🎙 Speech Signal Analysis using MFCC (Mel Frequency Cepstral Coefficients)

By integrating both modalities, the system enhances prediction reliability compared to single-mode approaches.

---

## 🚀 Key Features

### 🔐 Secure Authentication
- User Registration & Login (SQLite Database)
- Password hashing using `werkzeug.security`
- Session-based authentication

### ✍️ Handwriting Analysis
- Accepts 450 numerical feature inputs
- Random Forest Machine Learning Model
- Probability score calculation
- Disease stage estimation

### 🎙 Speech Analysis
- Accepts `.wav` and `.mp3` audio files
- Converts audio to:
  - Mono channel
  - 16kHz sampling rate
- Extracts 40 MFCC features
- Deep Learning model prediction

### 📊 Stage Estimation

Based on Alzheimer’s probability:

| Probability | Stage |
|------------|--------|
| < 40% | Healthy / No Significant Risk |
| 40% – 60% | Mild Cognitive Impairment (MCI) |
| 60% – 80% | Early Stage Alzheimer’s |
| > 80% | Advanced Alzheimer’s |

### 🛡 Dynamic Precautions

The system provides precautionary recommendations based on predicted stage:
- Brain exercises
- Physical activity
- Medical consultation
- Lifestyle changes

---

## 🏗 System Architecture

