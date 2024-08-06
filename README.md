# Disease Prediction using Machine Learning

## Overview

This project aims to predict diseases based on given symptoms using machine learning techniques. The dataset contains 132 symptoms and 42 possible diseases, with both training and testing data provided. The goal is to develop accurate models that can assist in early and precise disease diagnosis.

## Libraries Used

- pandas
- numpy
- matplotlib
- seaborn
- sklearn
- scipy
- collections

## Data

The dataset consists of two CSV files:
- **Training.csv**: Used to train the models.
- **Testing.csv**: Used to evaluate the models.

Each file contains 133 columns: 132 symptoms and 1 prognosis (disease).

## Analysis Steps

1. **Data Preprocessing**: Handling missing values and exploring the dataset.
2. **Model Training**: Training Support Vector Machine (SVM), Gaussian Naive Bayes (NB), and Random Forest (RF) models.
3. **Model Evaluation**: Evaluating the models using cross-validation and confusion matrices.
4. **Ensemble Prediction**: Combining the predictions of the three models for better accuracy.
5. **Disease Prediction Function**: Implementing a function to predict diseases based on input symptoms.

## Insights

- All three models (SVM, Gaussian NB, RF) achieved perfect accuracy on the provided dataset.
- The ensemble approach combining the predictions of the three models also achieved perfect accuracy.

## Conclusion

Machine learning models can effectively diagnose diseases based on symptoms. The ensemble approach enhances the accuracy and reliability of predictions.

## Requirements

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- scipy

## Usage

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/Disease-Prediction-ML.git
    ```

2. Navigate into the project directory:
    ```bash
    cd Disease-Prediction-ML
    ```

3. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

4. Run the Jupyter notebook to see the analysis and results:
    ```bash
    jupyter notebook Disease_Prediction.ipynb
    ```

