# BenignMalignantPredictor

*BenignMalignantPredictor* is a Python-based machine learning project that aims to predict whether a breast mass is benign or malignant based on features computed from a digitized image of a fine needle aspirate (FNA). The project utilizes a dataset with ten real-valued features computed for each cell nucleus, including characteristics such as radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension.

## Table of Contents
- [**About the Dataset**](#about-dataset)
- [**Installation**](#installation)
- [**Usage**](#usage)
- [**Model Training and Evaluation**](#model-training-and-evaluation)
- [**Prediction**](#prediction)
- [**Best Model Selection**](#best-model-selection)

## About the Dataset
The features in the dataset are derived from a 3-dimensional space described in a paper by K. P. Bennett and O. L. Mangasarian. The dataset includes both mean and standard error values, as well as "worst" or largest values for various features, resulting in a total of 30 features. The dataset is available through the UCI Machine Learning Repository.

**Attribute Information:**
1. ID number
2. Diagnosis (M = malignant, B = benign)
3-32) Ten real-valued features for each cell nucleus, including radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension.

**Class Distribution:**
- 357 benign
- 212 malignant

[UCI Machine Learning Repository - Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/mpurvesh007/BenignMalignantPredictor.git
   cd BenignMalignantPredictor
Install the required dependencies:
bash
Copy code
pip install pandas numpy matplotlib seaborn scikit-learn

## Usage
The project consists of a Jupyter notebook (or Python script) that can be executed step by step. The following sections explain various aspects of the code:

**Exploratory Data Analysis:**
The notebook starts with exploratory data analysis, including checking the dataset's size, removing duplicate entries, and visualizing missing values. Additionally, it provides a correlation matrix heatmap and box plots for each feature.

**Data Preprocessing:**
Label encoding is applied to the 'diagnosis' column, where 0 represents benign (B) and 1 represents malignant (M). The dataset is then split into independent features (X) and the dependent variable (Y).

**Model Training:**
The project utilizes logistic regression and support vector machine (SVM) classifiers for prediction. Both models are trained using the training dataset.

**Model Evaluation:**
The notebook evaluates the models using accuracy, confusion matrix, and classification report metrics. It includes training accuracy, model accuracy score, confusion matrix, and a detailed classification report.

**Hyperparameter Tuning**
Grid search is employed to find the best hyperparameters for both logistic regression and SVM models. The best models are then trained and evaluated with the optimized hyperparameters.

## Model Training and Evaluation
The logistic regression model is trained using the default hyperparameters, and its performance is evaluated. Hyperparameter tuning is then performed using grid search, and the model is re-evaluated with the optimized hyperparameters.

The SVM model follows a similar process, with hyperparameter tuning performed using grid search.

## Prediction
The notebook includes an example of how to make predictions on new data. It provides an input_data array with feature values, and the trained logistic regression model predicts whether the patient is likely to have cancer or not.

## Best Model Selection
The notebook determines the best-performing model by comparing the accuracy and other metrics of the default and optimized models for both logistic regression and SVM.
