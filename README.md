<!-- About The Project -->
### Username / discourseID
--------- dandi-p4Gx --------------

## About the Project
### Auto Insurance Fraud Detection

![auto insurance claims](https://blog.privy.id/wp-content/uploads/2022/11/shutterstock_720284965-1-300x173.jpg)

### Business Objective
This research goal is to build binary classifier model which are able to separate fraud transactions from non-fraud transactions. We present our conclusions based on an empirical study comparing different ML models and classification metrics.

### Business Metrics

**Precision** and **recall** should be chosen as the one of the evaluation metrics in classification models.

<p align="center">
<img src="https://miro.medium.com/max/824/1*xMl_wkMt42Hy8i84zs2WGg.png"
  alt="Size Limit comment in pull request about bundle size changes"
  width="500" height="300">
</p>

Precision is the rate of true positives divided by the sum of true positives and false positives. Recall is the number of true positives divided by the sum of true positives and false negatives.
A high recall indicates the model is able to classify relevant (positive) results without mislabeling them as irrelevant (negative). On the other hand, high precision indicates the model is able to returned positives predicted values is correctly positives with low irrelevant results (incorrectly positives).

Recall score with low False Negative and high False Positive Rate in AUC score should be parameter to select best model.

### Working with data

1. Data Preparation

<p align="center">
<img src="https://github.com/DandiMahendris/Auto-Insurance-Fraud-Detection/blob/main/pict/Preparetion%20Diagram.png"
  alt="Size Limit comment in pull request about bundle size changes"
  width="300" height="420">
</p>

Kaggle provided dataset of Auto Insurance Claim contains 1000 rows and 40 columns shape with unbalanced dataset (75%-25%).

2. Data Preprocessing and Feature Engineering

<p align="center">
<img src="https://github.com/DandiMahendris/Auto-Insurance-Fraud-Detection/blob/main/pict/Preprocessing%20Diagram.png"
  alt="Size Limit comment in pull request about bundle size changes"
  width="500" height="550">
</p>

Data separated into **predictor(X)** and **label(y)**. numerical and categorical type of data in predictor is splitted and both missing value handled.

3. Data Modelling

<p align="center">
<img src="https://github.com/DandiMahendris/Auto-Insurance-Fraud-Detection/blob/main/pict/Modelling%20Diagram.png"
  alt="Size Limit comment in pull request about bundle size changes"
  width="600" height="580">
</p>