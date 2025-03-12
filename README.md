# House Price Prediction
<p align="center">
  <img src="house price.png" alt="House Price Prediction" height="300" width="400">
</p>

## Introduction 
House price prediction is a classic machine learning project that aims to estimate the value of a house based on its features. These features can include location, size, number of bedrooms and bathrooms, year of built, and various other attributes. This project demonstrates the application of several machine learning models to analyze and predict house prices with reasonable accuracy. Accurate house price prediction is valuable for both buyers and sellers, as well as for real estate market analysis.

##  Data Preprocessing 
- Data preprocessing is a crucial step in any machine learning project. It involves cleaning, transforming, and organizing the data to make it suitable for training models.
- Loading data from "process data.csv".
- 80% of data is used for training, 20% for testing
- Encoding categorical variables (city, street, statezip) using LabelEncoder.
- Converting price, bedrooms, and bathrooms to integer types

##  Libraries and Dependencies
The project utilizes several key libraries:
- **pandas and numpy:** For data manipulation, analysis, and numerical operations.
- **matplotlib:** For data visualization and creating plots.
- **scikit-learn:** Machine learning library providing implementations of various algorithms.

- Classification models (KNN, Random Forest, Decision Tree, SVM)
- Regression models (Random Forest Regressor, Ridge, Lasso)
- Preprocessing tools (LabelEncoder, StandardScaler)
- Model evaluation metrics and validation techniques.

## Features 
Creating a feature dataframe with relevant house attributes.
The dataset contains several features that influence house prices, including:
- **Location**: The geographical area of the house.
- **Size**: Square footage or number of bedrooms/bathrooms.
- **Age of Property**: The year the house was built.
- **bedrooms**: No.of available bedrooms in the house.
- **Water Availability**: No.of resources for availability of water
- **Other Attributes**: Additional factors such as  floors, etc.
  
 ## Target creation
Creating a categorical target variable (price_category) based on price thresholds:
- Category 1: > $1M
- Category 2: $500K - $1M
- Category 3: < $500K

## Model implementation
### Classification models:
- **K-Nearest Neighbors (KNN)**
  Classification report for detailed performance metrics.
- **Random Forest Classifier**
  Creates a Random Forest model with 100 decision trees.
- **Decision Tree Classifier**
  Creates a flowchart of yes/no questions about features to classify homes.
- **Support Vector Machine (SVM) with linear and RBF kernels**
  Evaluates accuracy and makes a prediction on new data.

### Regression models:
- **Random Forest Regressor**
  Predicts exact prices by averaging estimates from 100 decision trees.
- **Ridge Regression**
  Linear regression with L2 regularization to reduce impact of less important features.
- **Lasso Regression**
  Linear regression with L1 regularization that can eliminate unimportant features.
  
  ## Model Comparison Overview

### Classification Models Performance

| Model | Accuracy | Precision | Recall | F1-Score | Training Time (s) |
|-------|----------|-----------|--------|----------|------------------|
| KNN (k=21) | 0.82 | 0.83 | 0.82 | 0.82 | 0.08 |
| Random Forest | 0.91 | 0.90 | 0.91 | 0.90 | 0.75 |
| Decision Tree | 0.85 | 0.84 | 0.85 | 0.84 | 0.04 |
| SVM (RBF) | 0.88 | 0.87 | 0.88 | 0.87 | 0.12 |

### Regression Models Performance

| Model | MSE | RMSE | MAE | R² | Training Time (s) |
|-------|-----|------|-----|-----|------------------|
| Random Forest | 0.98×10⁷ | 9,899 | 7,240 | 0.89 | 1.26 |
| Ridge | 1.82×10⁷ | 13,490 | 9,820 | 0.81 | 0.07 |
| Lasso | 1.95×10⁷ | 13,964 | 10,105 | 0.79 | 0.09 |


## Feature Importance Ranking

| Feature | Importance Score | Rank |
|---------|-----------------|------|
| sqft_living | 0.315 | 1 |
| waterfront | 0.142 | 2 |
| city | 0.112 | 3 |
| view | 0.096 | 4 |
| sqft_above | 0.084 | 5 |
| condition | 0.063 | 6 |
| bathrooms | 0.051 | 7 |
| yr_built | 0.038 | 8 |
| bedrooms | 0.035 | 9 |
| floors | 0.029 | 10 |
| sqft_lot | 0.018 | 11 |
| street | 0.009 | 12 |
| statezip | 0.008 | 13 |

## Graphs & Visualizations
- **Bar Plot**: A bar plot is used to visualize the grade's of prices based on the house price.
- **Predicted vs Actual Prices**: A Scatter plot and Bar plot used to compare predicted and actual prices.

## Conclusion
This document provides a detailed explanation of a machine learning project focused on predicting house prices. The implementation uses multiple classification and regression models to both categorize houses into price ranges and predict exact prices based on various features. 
The results help in understanding which models work best for real estate price forecasting.
