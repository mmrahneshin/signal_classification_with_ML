# Signal Classification with Machine Learning

This repository contains a comprehensive machine learning project designed to classify high-dimensional signal data. The project leverages advanced feature engineering techniques, a variety of machine learning algorithms, and visualization tools to analyze and evaluate the data effectively. It is a practical and modular framework, suitable for both research and real-world applications in signal processing and classification.

## Dataset Overview

- **Features (`x.pkl`)**: The dataset consists of 500 samples, each represented by 4097 numerical features. These features may correspond to time-series or signal data extracted from a specific domain.
- **Labels (`y.pkl`)**: A 1D array of 500 labels, with each label categorizing the corresponding sample for supervised learning tasks.

## Key Highlights

1. **Classification Models**: 
   - A diverse set of algorithms implemented, including:
     - **Support Vector Machine (SVM)**
     - **K-Nearest Neighbors (KNN)**
     - **Random Forest**
     - **Convolutional Neural Networks (CNN)** for deep learning-based classification.
   - Customizable training and evaluation pipelines for each model.

2. **Feature Engineering**:
   - Methods to process and transform raw signal data into meaningful features.
   - Includes both time-domain and frequency-domain features.
   - Statistical feature extraction to enhance model performance.

3. **Model Evaluation and Visualization**:
   - Tools to generate confusion matrices and ROC curves for assessing classification performance.
   - Implements K-fold cross-validation for robust evaluation of models.

4. **Clustering and Unsupervised Learning**:
   - Scripts for exploring the dataset through clustering methods, allowing for unsupervised insights.

## Repository Structure

- **`classification/`**: Contains scripts for training and testing classification models like SVM, KNN, and CNN.
- **`features/`**: Includes scripts for extracting features and preparing the dataset.
- **`chart/`**: Visualization tools for creating confusion matrices and ROC curves.
- **`doc/`**: Detailed documentation and experiment reports, including performance metrics and results.
- **`select_features/`**: Scripts for selecting the most relevant features to improve model performance.
- **`phase 3/`**: Advanced approaches, such as deep learning-based classification and feature optimization.

## Why This Project?

This project serves as a powerful resource for anyone interested in:
- Applying machine learning techniques to high-dimensional datasets.
- Understanding feature engineering for signal or time-series data.
- Experimenting with multiple classification algorithms in a modular framework.
- Exploring the application of deep learning in signal processing tasks.

