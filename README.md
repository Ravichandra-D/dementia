# Dementia MRI Classification Using Deep Learning with Synthetic Data Generation

This project presents a complete deep-learning pipeline for dementia classification using brain MRI scans, combined with synthetic data generation using GANs and VAEs to enhance dataset diversity and support privacy-preserving research. The system integrates multiple models (CNN, ResNet50, MobileNetV2) and deploys real-time prediction through a Streamlit interface.

##  Project Highlights
- MRI-based dementia classification across four classes  
  **(Non-Demented, Very Mild Demented, Mild Demented, Moderate Demented)**  
- Multiple deep learning models:  
  . Custom CNN  
  . ResNet50 (Transfer Learning)  
  . MobileNetV2 (Transfer Learning)  
- Synthetic MRI generation using:  
  . GAN (Generative Adversarial Network)  
  . VAE (Variational Autoencoder)  
- Robust validation using:  
  . Train–Test Swap  
  . Synthetic–Real Mixing Ratio Analysis  
  . Real Hold-Out Calibration  
- End-to-end deployment using **Streamlit**  
- Models trained using **Google Colab**, deployed via **VS Code**

## 📂 Dataset
This project uses a dementia MRI dataset containing **5,120 images** across four classes.  
Images were resized, normalized, and converted to grayscale (for CNN) and RGB (for ResNet50/MobileNetV2).

Source: HuggingFace — https://huggingface.co/datasets/falah/Alzheimer_MRI

##  Preprocessing Summary
- Convert raw pixel arrays to PIL images  
- Resize all MRIs to 128×128  
- Convert to grayscale for CNN and RGB for transfer-learning  
- Normalize pixel intensities → [0, 1]  
- Add channel dimension for CNN → (128,128,1)  
- One-hot encode labels  
- Train–test split = 80/20  
- Convert dataset using HuggingFace `.with_format("numpy")`  

## 🧱 Model Architectures

### 1️⃣ Custom CNN (Improved)
```
Input (128×128×1)
→ Conv2D → BN → Conv2D → BN → MaxPool → Dropout
→ Conv2D → BN → Conv2D → BN → MaxPool → Dropout
→ Conv2D → BN → Conv2D → BN → MaxPool → Dropout
→ Flatten → Dense(256) → Dropout
→ Softmax (4 classes)
```

### 2️⃣ ResNet50 (Transfer Learning)
```
Input (128×128×3)
→ Pretrained ResNet50
→ GAP
→ Dense(256)
→ Softmax (4 classes)
```

### 3️⃣ MobileNetV2 (Transfer Learning)
```
Input (128×128×3)
→ Pretrained MobileNetV2
→ GAP
→ Dense(128)
→ Softmax (4 classes)
```

### 4️⃣ GAN & VAE  
GAN generates realistic MRI images, while VAE learns latent representations and reconstructs MRIs.

##  Evaluation Metrics
Accuracy:
TP + TN / (TP + TN + FP + FN)

Precision:
TP / (TP + FP)

Recall:
TP / (TP + FN)

F1-score:
2 × (Precision × Recall) / (Precision + Recall)

Synthetic validation:
- Train–test swap  
- Mixing ratios (0–100% synthetic)  
- Real hold-out testing  

##  Results Summary
- Improved CNN ≈ 92% accuracy  
- ResNet50 most stable & accurate  
- MobileNetV2 high accuracy + lightweight  
- GAN & VAE generated realistic synthetic scans  
- Mixed synthetic training improved generalization  

## 🌐 Streamlit Deployment
Run:
```
streamlit run app.py
```

## 📁 Folder Structure
```
project/
│── app.py
│── cnn_model.h5
│── resnet_model.h5
│── mobilenet_model.h5
│── gan_generator.h5
│── vae_decoder.h5
│── sample_images/
│── README.md
```

## 📦 Installation
```
pip install tensorflow streamlit pillow numpy matplotlib opencv-python
```
Use Python 3.10 (TensorFlow does not support 3.12).

## Dataset
Alzheimer MRI Classification Dataset — HuggingFace. https://huggingface.co/datasets/falah/Alzheimer_MRI


## 📞 Contact
**Ravichandra D**  
Email: ravichandra182001@gmail.com
