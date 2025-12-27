# Skin Cancer Detection Model using Deep Learning & Hybrid CNN-Transformer

An automated diagnostic system developed for **CSE 445: Machine Learning** at **North South University**. This project implements a multi-model approach to classify skin cancer from dermoscopic images into five distinct categories.


## 🔬 Project Overview
This project addresses the critical need for early skin cancer detection by targeting five specific types of lesions: Actinic Keratoses, Basal Cell Carcinoma, Benign Keratosis-like Lesions, Melanocytic Nevi, and Melanoma. We addressed severe class imbalance and high inter-class visual similarity using a rigorous data processing pipeline.

---

## 🏗️ Model Architectures

### 1. Custom Hybrid CNN-Transformer
* **Layers**: 69 layers with 12 million trainable parameters.
* **Design**: Fuses Convolutional, Residual, and MBConv blocks with a Multi-head self-attention Transformer block at the bottleneck to capture global context.

### 2. ResNet152 (Transfer Learning)
* **Layers**: 152 layers with ~795 million parameters (354 million trainable).
* **Modification**: Custom classification head with a Dropout layer (0.15) and a linear layer for 5 output classes.

---

## 🧪 Methodology
* **Preprocessing**: Images resized to $224 \times 224$ and denoised via `cv2.fastNlMeansDenoisingColored` to remove artifacts like hair.
* **Augmentation**: Used the `albumentations` library for rotation, flips, and color jitter to balance minority classes.
* **Optimization**: Trained using the Adam Optimizer and Cross-Entropy Loss with early stopping and `ReduceLROnPlateau`.
* **Ensemble**: A soft-voting strategy combines predictions from both models to improve final diagnostic accuracy.

---

## 📊 Performance Results
The models were evaluated on a test set of 1,253 images.

| Model | Test Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **ResNet152** | 94.21% | 0.9368 | 0.9284 | 0.9326 |
| **Hybrid CNN-Transformer** | 92.58% | 0.9333 | 0.9258 | 0.9281 |
| **Ensemble (Final)** | **~95.00%** | 0.9333 | 0.9258 | 0.9223 |



---


### Team Members
* **Hasnat Karibul Islam** - NSU ID: 2211275042
* **Mahin Ahmed** - NSU ID: 2211772042

---


## 🙏 Acknowledgments
We express our sincere gratitude to our honorable faculty, **Dr. Mohammad Mahmudul Alam**, for providing the opportunity to work on this project and for his invaluable guidance throughout its development. 

We also thank our project-mates for their collaborative efforts and dedication to building this robust skin cancer detection model.




## 🛠️ Tools & Libraries
* **Language**: Python
* **Frameworks**: PyTorch, Keras, TensorFlow
* **Image Processing**: OpenCV, Albumentations
* **Analysis**: NumPy, Matplotlib, Scikit-learn

---
