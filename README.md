<!-- About The Project -->

<a name="readme-top"></a>

<!-- TABLE OF CONTENTS -->

<summary>Table of Contents</summary>
<ol>
  <li><a href="#about-the-project">About The Project</a></li>
  <li><a href="#business-objective">Business Objective</a></li>
  <li><a href="#business-metrics">Business Metrics</a></li>
  <li><a href="#getting-started">Getting Started</a></li>
  <li>
    <a href="#data-workflow">Data Workflow</a>
    <ul>
      <li><a href="#data-preparation">Data Preparation</a></li>
      <li><a href="#data-preprocessing-and-feature-engineering">Data Preprocessing and Feature Engineering</a></li>
<li><a href="#data-modelling">Data Modelling</a></li>      
    </ul>
  </li>
  <li>
    <a href="#prediction-using-api-and-streamlit">Prediction using API and Streamlit</a>
    <ul>
      <li><a href="#how-to-run-by-api?">How To Run by API?</a></li>
      <li><a href="#data-input">Data Input</a></li>
    </ul>
  </li>
</ol>

<!-- About the Project -->
## About the Project
## Auto Insurance Fraud Detection

<p align=center>
<img src="https://www.paulasmithinsurance.com/wp-content/uploads/2021/05/car-insurance-fraud-things-to-know-YI-1536x1024.jpg"
  width=550 height=320>
</p>

Frauds are known to be dynamic and have no patterns, hence they are not easy to identify. Amount claimed by fraudulent is significantly huge that may lead to serious loss of money. Such frauds occurred in all areas of insurance claim with high severity including auto sectors that widely claimed by fake accident claim. 

The project applying multiple ML techniques to label and classify fraud claim. Before applying any ML model, we use exploratory data analysis to find how the categoric and numeric features relationship toward target (fraud classification), as well as **feature selection** are used such as ***Chi-squared test*, *Shapiro-Wilk*, *Probability Plot, Welch's Test, Person Correlation, Lasso* Method, *XGBoost* feature importance** and ***Random Forest* feature importance**.

In preprocessing technique, dataset encountered with unpleasant task of high **imbalanced data** of fraud and non-fraud case. Hence, Oversampling and SMOTENC are applied toward dataset, and we will have 3 type of configuration data such as **Non-Balanced, *Oversampling*** and ***SMOTENC***.

The datasets using Auto Insurance Industry and will approached the ***accuracy*, *precision*, *recall*** on ***confusion* *matrix*** and also the ***Area Under the ROC Curve (AUC)*** as the evaluation metrics that would be used.

<!-- Business Objective -->
## Business Objective
This research goal is to build binary classifier model which are able to separate fraud transactions from non-fraud transactions. We present our conclusions based on an empirical study comparing different ML models and classification metrics.

<!-- Business Metrics -->
## Business Metrics

**Precision** and **recall** should be chosen as the one of the evaluation metrics in classification models.

<p align=center>
<img src=https://miro.medium.com/max/824/1*xMl_wkMt42Hy8i84zs2WGg.png
  alt=Size Limit comment in pull request about bundle size changes
  width=500 height=300>
</p>

**`Precision`** is the rate of true positives divided by the sum of true positives and false positives. **`Recall`** is the number of true positives divided by the sum of true positives and false negatives.
A high recall indicates the model is able to classify relevant (positive) results without mislabeling them as irrelevant (negative). On the other hand, high precision indicates the model is able to returned positives predicted values is correctly positives with low irrelevant results (incorrectly positives).

Recall score with low False Negative and high False Positive Rate in AUC score should be parameter to select best model.

<p align=center>
<img src=https://www.mathworks.com/help//examples/nnet/win64/CompareDeepLearningModelsUsingROCCurvesExample_01.png
  alt=Size Limit comment in pull request about bundle size changes
  width=400 height=350>
</p>

**AUC Score** is also another consideration to choose the best model. <br/>
**`ROC`** is an evaluation metric for binary classification problems and a probability curve that plots the **TPR** againts **FPR** at various threshold values. <br/>
An excellent model has AUC near to the 1 which means it has a good measure of separability. A poor model has an AUC near 0 which means it has the worst measure of separability

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

1. Clone the repository
```sh
$ git clone https://github.com/DandiMahendris/Auto-Insurance-Fraud-Detection.git`
```
2. Install requirement library and package on `requirements.txt`

3. If you want to create the folder instead

```sh
$ mkdir folder
$ git init
```

4. remote the repository
```sh
$ git remote add origin git@github.com:DandiMahendris/Auto-Insurance-Fraud-Detection.git
$ git pull origin main
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Data Workflow -->
# Data Workflow

## Dataset

Kaggle provided dataset of Auto Insurance Claim contains 1000 rows and 40 columns shape.

## Data Preparation

<p align=center>
<img src="https://github.com/DandiMahendris/Auto-Insurance-Fraud-Detection/blob/main/pict/Preparetion%20Diagram.png"
  width=350 height=420>
</p>

## Data Preprocessing and Feature Engineering

<p align=center>
<img src="https://github.com/DandiMahendris/Auto-Insurance-Fraud-Detection/blob/main/pict/Preprocessing%20Diagram.png"
  width=500 height=550>
</p>

Data separated into **predictor(X)** and **label(y)**. Numerical and categorical type of data in predictor is splitted and both missing value is handled using `SimpleImputer`

```
SimpleImputer(missing_values = np.nan,
                                strategy = median)
```

```
SimpleImputer(missing_values = np.nan,
                                strategy = 'constant',
                                fill_value = 'UNKNOWN')
```

And for categorical data using `OneHotEncoder` and `OrdinalEncoder` from sklearn package.
```
OneHotEncoder(handle_unknown = 'ignore',
                                drop = 'if_binary')
```
```
OrdinalEncoder(categories=[incident_type,witnesses,incident_severity,auto_year,
                                   umbrella_limit,bodily_injuries,number_of_vehicles_involved])
```

Next, cat and num data is concatenated to normalize the data. normalization method uses `standardscaler` from sklearn package.

This normalized data is concatenated with label(y) to be balanced using `SMOTE` and `Oversampling`. However, to capture benchmark of multiple ML models, we also use unbalanced dataset as **nonbalancing**.

## Data Modelling

<p align=center>
<img src=https://github.com/DandiMahendris/Auto-Insurance-Fraud-Detection/blob/main/pict/Modelling%20Diagram.png
  alt=Size Limit comment in pull request about bundle size changes
  width=680 height=580>
</p>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Prediction Using API and Streamlit -->

## Prediction using API and Streamlit
### How To Run by API?

> 1. Make sure you have **Git** installed as well. If not, search for "How to install git on Windows/Ubuntu/Os that you used."

> 2. Make a clone of this repository or download files on this repository:
`git clone https://github.com/DandiMahendris/Auto-Insurance-Fraud-Detection.git` 

> 3. Open a CMD terminal and navigate to the cloned folder's directory. Try to test API by following the code:
`python .\src\api.py`

> 4. To try streamlit. Open CMD terminal and type the code:
`streamlit run .\src\streamlit.py`

### Data Input

Input Api uses json format like this:
```JSON
{
  "injury_claim": int,
  "property_claim": int,
  "vehicle_claim": int,
  "insured_hobbies": str,
  "incident_type": str,
  "collision_type": str,
  "incident_severity": str,
  "authorities_contacted": str,
  "incident_state": str,
  "property_damage": str
}
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>