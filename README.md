# Airline Customer Satisfaction Analysis

This project analyzes airline customer satisfaction data using various machine learning models and data visualization techniques to predict passenger satisfaction. The dataset is analyzed for key factors that impact satisfaction, including service-related variables, flight details, and demographic information.

## Table of Contents
1. [Overview](#overview)
2. [Setup](#setup)
3. [Data Exploration](#data-exploration)
4. [Data Visualization](#data-visualization)
5. [Correlation Analysis](#correlation-analysis)
6. [Modeling](#modeling)
   - Logistic Regression
   - K-Nearest Neighbors (KNN)
   - Naive Bayes
   - Decision Tree
   - Random Forest
7. [Results and Evaluation](#results-and-evaluation)
8. [Conclusion](#conclusion)

## Overview
This project uses machine learning models to predict passenger satisfaction (`Sat`) based on various factors like seat comfort, food quality, flight delay times, and passenger demographics. The dataset is split into training and test sets, and several models are trained and evaluated for predictive performance.

## Setup

To run the analysis, follow these steps:

1. **Install Required Packages**  
   The following R packages are required for data manipulation, visualization, and modeling:
   - `caret`
   - `randomForest`
   - `kknn`
   - `naivebayes`
   - `glmnet`
   - `pROC`
   - `PRROC`
   - `ROSE`
   - `reshape2`
   - `psych`
   - `dplyr`
   - `rpart`
   - `class`
   - `rpart.plot`
   - `pander`
   - `nortest`
   - `ggplot2`

   To install missing packages, the script will automatically install them.

2. **Set Working Directory and Load Data**  
   The script prompts you to choose the CSV file containing the dataset, which will be loaded into the variable `airlinedata`.

3. **Load and Install Dependencies**  
   If you haven't already installed the required packages, the script will install them automatically.

## Data Exploration
The dataset is initially explored using the following functions:
- `head(airlinedata)`: View the first few rows of the dataset.
- `str(airlinedata)`: Display the structure of the data.
- `summary(airlinedata)`: Get summary statistics of the dataset.

### Categorical Variables
The script defines a list of categorical variables and converts them to factors for modeling.

## Data Visualization

### Key Visualizations:
1. **Gender Distribution**: Bar plot showing the count of male and female passengers.
2. **Customer Type Distribution**: Bar plot depicting the distribution of loyal vs. disloyal customers.
3. **Age Distribution**: Histogram representing the age distribution of passengers.
4. **Satisfaction Distribution**: Bar plot of the satisfaction levels (Dissatisfied vs. Satisfied).
5. **Satisfaction vs. Services**: Box plots comparing satisfaction scores across different services (e.g., seat comfort, food drink, cleanliness).

## Correlation Analysis
The script calculates the correlation matrix for the numeric variables in the dataset and visualizes the correlations using a heatmap. Categorical variables are converted to numeric before calculating the correlation matrix.

## Modeling

### Logistic Regression
The script trains two logistic regression models to predict passenger satisfaction (`Sat`), using various predictor variables. The model performance is evaluated using confusion matrices for both training and testing sets.

### K-Nearest Neighbors (KNN)
The script scales the predictor variables and trains two KNN models with different values of `k`. The performance of the models is evaluated using confusion matrices.

### Naive Bayes
The Naive Bayes classifier is trained on the dataset, and its performance is evaluated using a confusion matrix.

### Decision Tree
A decision tree model is trained to predict passenger satisfaction, and its performance is evaluated with a confusion matrix. The decision tree is also visualized.

### Random Forest
The Random Forest model is trained on the dataset, and its performance is evaluated with a confusion matrix.

## Results and Evaluation

After training each model, the performance metrics such as Accuracy, Sensitivity, Specificity, Precision, and Negative Predictive Value are visualized using bar plots. These metrics help to evaluate the effectiveness of each model.

## Conclusion
The project applies several machine learning techniques to predict passenger satisfaction based on various features. The models are compared based on their predictive accuracy and other evaluation metrics. The findings can help airlines improve customer satisfaction by focusing on the most significant factors influencing passenger experience.

## Files
- `airlinedata.csv`: Dataset containing information about passengers, flight details, and satisfaction ratings.
- `analysis_script.R`: The R script containing all code for data exploration, visualization, and modeling.
