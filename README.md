# Pneumonia Detection from Chest X-Rays using Deep Learning

## Project Overview
This computer vision project focuses on the classification of chest X-ray images to assist in the clinical diagnosis of pneumonia. The primary objective is clinical safety; therefore, the model is calibrated to maximize Recall (Sensitivity), strictly minimizing False Negative predictions to ensure infected patients are not dismissed without treatment.

## Technologies and Frameworks
* **Language:** Python
* **Deep Learning Framework:** PyTorch, Torchvision
* **Architecture:** ResNet18 (Transfer Learning)
* **Data Manipulation and Visualization:** Matplotlib, Seaborn, Scikit-learn, NumPy

## Methodology

1. **Exploratory Data Analysis (EDA):** Analyzed the pixel brightness distribution across the dataset to mathematically validate the opacity differences between normal and infected lungs.
2. **Preprocessing and Data Augmentation:** Applied standardized resizing (224x224), ImageNet-based normalization, and random rotations to the training set to mitigate overfitting.
3. **Transfer Learning:** Implemented a pre-trained ResNet18 architecture. The convolutional base (feature extractors) was frozen to optimize GPU processing time, while the final dense layer (Fully Connected) was replaced and trained specifically for this binary classification task.
4. **Optimization:** Utilized the Adam optimizer and CrossEntropyLoss function.

## Results and Clinical Threshold Tuning

In medical diagnostics, a False Negative (classifying an infected patient as healthy) carries significantly more risk than a False Positive. 

Due to natural dataset imbalance, the baseline model output an unacceptable rate of False Negatives (180 cases). To align the model with healthcare safety standards, a threshold tuning procedure was applied. The decision boundary was lowered from the default `0.50` to `0.30`.

**Final Metrics (Threshold = 0.30):**
* **Global Accuracy:** 83.33%
* **Recall / Sensitivity:** 99.23%
* **Specificity:** 56.83%
* **False Negatives:** Reduced from 180 to 3 cases.

This calibration demonstrates the model's viability as a highly sensitive triage tool for clinical environments.
