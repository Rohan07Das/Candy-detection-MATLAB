# 🍬 Candy Detection and Classification using YOLOv4 and SqueezeNet (MATLAB)

This project presents an object detection and classification system for identifying different types of candies using deep learning techniques. The system combines **YOLOv4** for object detection and **SqueezeNet** for classification, all developed and evaluated within **MATLAB R2023a**.

---

## 📖 Project Overview

The goal of this project is to detect and classify multiple types of candies from images using computer vision and deep learning. A total of **11 custom candy classes** were labeled and used to train two models:

- A **YOLOv4** object detector with a CSPDarknet53 backbone for identifying the location of candies in images.
- A **SqueezeNet** classifier to determine the exact type of candy detected in each region.

The system is capable of:
- Detecting multiple candy types in a single image
- Classifying each detected candy accurately
- Running in MATLAB and Simulink for real-time applications

---

## 📂 Project Structure

├── models/ # Trained YOLOv4 and SqueezeNet models
├── src/ # MATLAB scripts and Simulink model files
├── datas/ # Input images and class annotations
├── results/ # Sample outputs, evaluation plots
├── README.md # Project overview and instructions

## 🧠 Key Components

- **YOLOv4 Detector**  
  Detects bounding boxes for candies from input images using CSPDarknet53.

- **SqueezeNet Classifier**  
  Classifies detected candy regions into one of 11 predefined classes.

- **MATLAB Simulink Integration**  
  Designed for real-time processing and future embedded deployment.

---

## 📊 Results

- Successfully detected and classified 11 candy types.
- Achieved high classification accuracy with SqueezeNet.
- Real-time testing in MATLAB with Simulink integration.

---

## 📦 Download Required Files(these are all come under the model files)

Due to GitHub's file size restrictions, large files are hosted on Google Drive. Download and place them appropriately:

| File/Folder                          | Description                              | Download Link |
|--------------------------------------|------------------------------------------|----------------|
| 📁 Dataset Folder (`/datas/`)         | Candy image dataset (11 classes)         | [Open Folder](https://drive.google.com/drive/folders/1RHsn0Fgs0_e_5mvP3-qHxQ-RYlvHUPnO?usp=drive_link) |
| 🧠 SqueezeNet Classifier (`SqueezeNetClassifier.mat`) | Trained classification model | [Download](https://drive.google.com/uc?export=download&id=17_YW1-KpLukBzGVkkF4s8jpArsPjX_17) |
| 🤖 YOLOv4 Model (`YOLOv4Candy.mat`)  | Trained object detector model            | [Download](https://drive.google.com/uc?export=download&id=1Ov5-v72LhNlKgNNvpbii3OEPGRGc_2QC) |

> ⚠️ After downloading:
> - Place `.mat` files in the `models/` folder
> - Copy dataset into `datas/` folder as-is

