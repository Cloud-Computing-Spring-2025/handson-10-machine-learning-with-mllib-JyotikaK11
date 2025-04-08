# handson-10-MachineLearning-with-MLlib.

# Customer Churn Prediction with MLlib

This project uses Apache Spark MLlib to predict customer churn based on structured customer data. You will preprocess data, train classification models, perform feature selection, and tune hyperparameters using cross-validation.

---

Build and compare machine learning models using PySpark to predict whether a customer will churn based on their service usage and subscription features.

---

## Dataset

The dataset used is `customer_churn.csv`, which includes features like:

- `gender`, `SeniorCitizen`, `tenure`, `PhoneService`, `InternetService`, `MonthlyCharges`, `TotalCharges`, `Churn` (label), etc.

---

## Tasks

### Task 1: Data Preprocessing and Feature Engineering

**Objective:**  
Clean the dataset and prepare features for ML algorithms.

**Steps:**
1. Fill missing values in `TotalCharges` with 0.
2. Encode categorical features using `StringIndexer` and `OneHotEncoder`.
3. Assemble numeric and encoded features into a single feature vector with `VectorAssembler`.

**Code Output:**
```
+--------------------+-----------+
|features            |ChurnIndex |
+--------------------+-----------+
|[0.0,3.0,94.8,267.3,|        0.0 |
|(8,[1,2,3,6],[29.0,7|        0.0 |
|(8,[1,2,3,6],[15.0,9|        0.0 |
|(8,[1,2,3,4],[53.0,3|        0.0 |
|(8,[2,5,6],[83.35,1.|        0.0 |
+--------------------+-----------+
```

---

### Task 2: Train and Evaluate Logistic Regression Model

**Objective:**  
Train a logistic regression model and evaluate it using AUC (Area Under ROC Curve).

**Steps:**
1. Split dataset into training and test sets (80/20).
2. Train a logistic regression model.
3. Use `BinaryClassificationEvaluator` to evaluate.

**Code Output:**
```
Sample Predictions:
+-------+-----------+------------------------------------------+
|Label  |Prediction |Probability                               |
+-------+-----------+------------------------------------------+
|   0.0 |         0 |[0.5919130918951714,0.4080869081048286] |
|   1.0 |         0 |[0.8440056167635827,0.15599438323641734]|
|   0.0 |         0 |[0.8979433651527896,0.10205663484721039]|
|   1.0 |         1 |[0.3367126294222577,0.6632873705777422] |
|   0.0 |         0 |[0.5395984060468031,0.4604015939531969] |
+-------+-----------+------------------------------------------+

Logistic Regression Model Accuracy (AUC): 0.75
```

---

### Task 3: Feature Selection using Chi-Square Test

**Objective:**  
Select the top 5 most important features using Chi-Square feature selection.

**Steps:**
1. Use `ChiSqSelector` to rank and select top 5 features.
2. Print the selected feature vectors.

**Code Output:**
```
+--------------------+-----------+
|selectedFeatures    |ChurnIndex |
+--------------------+-----------+
|[0.0,3.0,1.0,0.0,1.0|        0.0 |
|(5,[1,3],[29.0,1.0])|        0.0 |
|(5,[1,3],[15.0,1.0])|        0.0 |
|(5,[1],[53.0])      |        0.0 |
|(5,[2,3],[1.0,1.0]) |        0.0 |
+--------------------+-----------+
```

---

### Task 4: Hyperparameter Tuning and Model Comparison

**Objective:**  
Use CrossValidator to tune models and compare their AUC performance.

**Models Used:**
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosted Trees (GBT)

**Steps:**
1. Define models and parameter grids.
2. Use `CrossValidator` for 5-fold cross-validation.
3. Evaluate and print best model results.

**Code Output:**
```
Tuning LogisticRegression...
LogisticRegression Best Model Accuracy (AUC): 0.75
Best Params for LogisticRegression:
  maxIter = 100
  regParam = 0.01

Tuning DecisionTree...
DecisionTree Best Model Accuracy (AUC): 0.80
Best Params for DecisionTree:
  maxDepth = 5

Tuning RandomForest...
RandomForest Best Model Accuracy (AUC): 0.78
Best Params for RandomForest:
  maxDepth = 5
  numTrees = 10

Tuning GBT...
GBT Best Model Accuracy (AUC): 0.80
Best Params for GBT:
  maxDepth = 5
  maxIter = 20
```

---

### 🏆 Final Model Selection

```
Best Model Overall: DecisionTree with AUC = 0.80
Best Hyperparameters:
  maxDepth = 5
```

---

## Execution Instructions

### 1. Prerequisites

- Apache Spark installed
- Python environment with `pyspark` installed
- `customer_churn.csv` placed in the project directory

### 2. Run the Project

```bash
spark-submit customer-churn-analysis.py
```

### 3. View Results

```bash
cat output/final_output.txt
```

All outputs from all tasks will be written to this file.

---

## Notes

- AUC is used as the evaluation metric across all models.
- Chi-Square is used for feature selection.
- Gradient Boosted Trees and Decision Trees achieved the highest AUC (0.80), but the best-performing model overall was **Decision Tree** based on its interpretability and stability.
