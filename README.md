# <center> Crop-Disease-Detection</center>
### 🌱 From Pixels to Diagnosis: The Crop Guardian AI 🌱
Welcome to the Crop Disease Detection project! This repository contains the code for an intelligent system that uses deep learning to identify diseases in plant leaves, helping farmers and botanists protect their crops and increase yields.

<img width="2164" height="2114" alt="image" src="https://github.com/user-attachments/assets/967dbf9f-2f60-41d0-aa23-dbdbe95b03b3" />

Have you ever wondered if an AI could tell a healthy leaf from a sick one just by looking at it? That's what this project is all about. We've built a powerful image classification model that can instantly recognize and diagnose common plant diseases.

### 💻 Project Highlights

<b>Dataset</b> : PlantVillage-Dataset ➡️ https://github.com/spMohanty/PlantVillage-Dataset

<b>Model</b>: We've used a Convolutional Neural Network (CNN), the go-to architecture for image classification. The notebook includes a custom-built CNN as well as a more advanced transfer learning approach using a pre-trained EfficientNetB0 model.

<b>Data Augmentation: To make our model more robust, we've applied extensive data augmentation techniques. This means we artificially expanded our dataset by rotating, flipping, and zooming into images, teaching the model to recognize diseases from all angles.

<b>Technologies: This project is built using a powerful stack of Python libraries.</b>

<b>TensorFlow & Keras: The core of our deep learning model.</b>

<b>Numpy & Matplotlib: For numerical operations and visualizing our model's performance.</b>

<b>Scikit-learn: For generating classification reports and confusion matrices.</b>

<b>ImageDataGenerator: Our workhorse for efficient data preprocessing and loading.</b>

---

### 📉 Model Performance & Evaluation
<img width="400" alt="PlantDisease1" src="https://github.com/user-attachments/assets/035caa96-f9df-4834-9db8-bcadd172065d" />
<img width="400" alt="Plantdisease2" src="https://github.com/user-attachments/assets/85be54d2-8982-4625-92bb-8f75f8b0310b" />

<b>At last epoch these are the metrics ⏬ </b>
<table>
  <tr><th>accuracy</th><td> 0.8125 </td></tr>
  <tr><th> loss</th><td> 0.5851 </td></tr>
  <tr><th> val_accuracy</th><td> 0.8138 </td></tr>
  <tr><th> val_loss</th><td> 0.5858</td></tr>
</table>

---
### 🚀 Getting Started
Ready to explore the code? The heart of this project is the CropDiseaseDetection.ipynb Jupyter Notebook.

To download the model : https://drive.google.com/file/d/1wo2emgwSwMDEUv4svqiNhw1clWtKaBf8/view?usp=sharing

To run the project locally:

Clone the Repository:

```Bash

git clone https://github.com/ShubhamP1028/Crop-Disease-Detection.git
```

Install Dependencies:

```Bash
pip install -r requirements.txt
```
