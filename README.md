# Breast_Cancer-Classification-using-Neural_Networks

##  Project Overview

This project implements an **Artificial Neural Network (ANN)** to classify breast cancer tumors as **Malignant** or **Benign** using the **Wisconsin Breast Cancer (Diagnostic) Dataset**. The model is built with **TensorFlow/Keras** and demonstrates a complete machine learning pipeline from preprocessing to evaluation.

---

##  Objectives

* Classify breast cancer tumors accurately
* Apply deep learning for medical data analysis
* Perform data preprocessing and feature scaling
* Evaluate model performance using metrics and visualizations

---

## Dataset Description

**Dataset:** Breast Cancer Wisconsin (Diagnostic)

* **Features:** 30 numerical features extracted from FNA images (radius, texture, perimeter, area, smoothness, etc.)
* **Target Labels:**

  * `0` → Malignant (Cancerous)
  * `1` → Benign (Non-cancerous)

---

## Tech Stack

* **Programming Language:** Python
* **Deep Learning:** TensorFlow (Keras)
* **Data Handling:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Preprocessing:** Scikit-Learn (StandardScaler, train_test_split)

---

##  Workflow

1. Load dataset (sklearn / CSV)
2. Handle missing values (if any)
3. Encode target labels
4. Scale features using StandardScaler
5. Build ANN model
6. Train the model
7. Evaluate accuracy and confusion matrix

---

##  Model Architecture

```python
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(30,)),
    keras.layers.Dense(20, activation='relu'),
    keras.layers.Dense(2, activation='sigmoid')
])
```

* **Input Layer:** 30 features
* **Hidden Layer:** Dense (ReLU)
* **Output Layer:** Sigmoid (Binary Classification)

---

##  Training Details

* **Optimizer:** Adam
* **Loss Function:** Binary Crossentropy
* **Metric:** Accuracy

---

##  Results

* **Training Accuracy:** ~96–98%
* **Test Accuracy:** ~95–97%

> Results may vary slightly due to random initialization and data splits.

---

## Visualizations

* Accuracy vs Loss curves
* Confusion Matrix
* Custom prediction system for new inputs

---

##  Dependencies

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn
```

---



