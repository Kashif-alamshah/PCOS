🧫 PCOS Detection and Health Advisory using Deep Learning
This repository contains a full pipeline for detecting Polycystic Ovary Syndrome (PCOS) using ultrasound images with a deep learning model and providing personalized health suggestions through an integrated AI assistant powered by LangChain.

📽️ Project Overview
This project was built as part of my application for the Data & AI Intern position at MarketLytics.

🔍 Problem: Diagnosing PCOS through ultrasound imaging is time-consuming and can vary by radiologist interpretation.

🧠 Solution: A MobileNetV2-based CNN model that classifies ultrasound images as Infected (PCOS) or Non-Infected.

🤖 Bonus: Integrated LangChain-based AI assistant that gives practical lifestyle and health suggestions based on classification results.

🛠️ Tech Stack

Area	Tools/Libraries
Deep Learning	TensorFlow / Keras
Image Processing	OpenCV, Matplotlib, Seaborn
Web App	Streamlit
AI Assistant	LangChain + HuggingFace Hub (TinyLlama)
Deployment	Localhost or Cloud (Streamlit sharing)
🚀 How It Works
Data Preparation:

Ultrasound images are split into infected and non-infected classes.

Images are augmented and normalized using ImageDataGenerator.

Model Architecture:

Transfer learning with MobileNetV2.

Additional Dense + Dropout layers.

Optimized using class weights and early stopping.

Evaluation:

Precision-Recall curve plotted.

Optimal classification threshold is selected based on F1-score.

Confusion matrix + Classification report.

Streamlit App:

Upload an ultrasound image.

Model predicts infection status.

AI assistant provides health guidance based on result.

🧠 Demo Preview
🖼️ Model Prediction
Upload ultrasound → Model classifies as Infected or Non-Infected → Results shown with color-coded response.

💡 AI Health Tips
LangChain generates tailored advice like diet suggestions, exercise tips, and hormonal health practices.
