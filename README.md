# 🧬 White Blood Cell Classification & Morphological Feature Extraction  
**Assignment 1 – Introduction to Deep Convolutional Neural Networks**  
**Goal: White Blood Cell (WBC) Classification + Morphology Detection**

---

## 📌 Project Overview  
This project implements a **Deep Convolutional Neural Network (CNN)** capable of performing two tasks:

### **1️⃣ Classifying White Blood Cells (WBCs)**  
The model predicts the WBC type (e.g., neutrophil, eosinophil, lymphocyte, monocyte).

### **2️⃣ Extracting Key Morphological Attributes**  
The system also predicts clinically relevant features such as:
- Cell Shape  
- Nucleus Shape  
- Cytoplasm Vacuoles  

This project uses **TensorFlow 2 with Keras** and follows a **multi-output CNN architecture**.  
Training and evaluation are done in a Jupyter Notebook environment, optimized to run both locally and on AWS.

---

## 🚀 Features  
- Multi-task Deep CNN (classification + morphology)  
- End-to-end training pipeline  
- Image preprocessing using OpenCV & TensorFlow  
- Data augmentation for better generalization  
- Evaluation using accuracy, F1-score, confusion matrix  
- TensorFlow 2.12 (Keras) workflow  
- AWS-friendly environment setup  

---

## 🛠️ Tech Stack  
- TensorFlow 2.12  
- Python 3.9  
- OpenCV  
- NumPy & Pandas  
- Matplotlib & Seaborn  
- Scikit-learn  
- JupyterLab  

---

## 📁 Project Structure  
```
WBC-MultiTask-CNN/
│
├── data/
│   ├── raw/               # Original images
│   ├── processed/         # Preprocessed images
│   └── labels.csv         # Label file
│
├── notebooks/
│   └── assignment_1.ipynb # Main training & evaluation notebook
│
├── models/
│   └── wbc_cnn.h5         # Saved trained model
│
├── environment.yml        # Conda environment file
├── requirements.txt       # Package dependencies
├── README.md              # Project documentation
└── .gitignore
```

---

## 🔧 Installation & Setup  

### **1️⃣ Clone this repository**
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### **2️⃣ Create the Conda environment**
```bash
conda env create -f environment.yml
conda activate wbc-cnn-env
```

### **3️⃣ Launch JupyterLab**
```bash
jupyter lab
```

---

## 🧪 Training the Model  
Open the notebook:

```
notebooks/main.ipynb
```

Inside the notebook, follow the guided steps:
- Load dataset  
- Preprocess images  
- Build the CNN architecture  
- Configure multi-output losses  
- Train the model  
- Evaluate the results  

---

## 📊 Evaluation  
Model performance is measured using:

- **WBC Classification Accuracy**  
- **Morphological Feature Accuracy**  
- **F1-score**  
- **Confusion Matrix**  
- **Training & validation curves**

These metrics assess generalization to unseen microscopic images.

---

## ☁ Deployment (AWS Ready)  
This setup is compatible with:
- AWS EC2  
- AWS Sagemaker Notebook Instances  
- GPU-enabled EC2 instances  

---

## 📌 Future Improvements  
- Expand dataset for more morphology labels  
- Experiment with deeper CNN designs  
- Use multi-branch CNNs for morphology prediction  
- Add deployment interface (FastAPI / Streamlit)

---

## 📜 License  
This project is for academic use under RMIT University (Assignment 1).

---
```
