# Campaign Conversion Prediction Model

## Overview

This project demonstrates a Python-based machine learning solution for predicting customer campaign conversion likelihood using behavioural analytics, demographic segmentation, and financial engagement indicators.

The model was developed using Logistic Regression and simulates a real-world financial services marketing propensity modelling workflow for products including:

- CTF
- JISA
- ISA
- LISA

The solution predicts future customer conversion probability, prioritises campaign customers, and supports outbound marketing optimisation.

---

# Business Objective

The objective of this project is to identify customers most likely to convert during outbound marketing campaigns by analysing:

- Customer engagement behaviour
- Financial product maturity
- Direct Debit activity
- Website engagement
- Email response behaviour
- Customer demographics
- CACI Fresco-style segmentation

The model helps marketing teams prioritise customers and improve campaign ROI.

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Jupyter Notebook
- Joblib
- Logistic Regression
- Machine Learning Pipelines

---

# Machine Learning Workflow

The project demonstrates a complete end-to-end machine learning workflow including:

1. Historical dataset preparation
2. Feature engineering
3. Target variable creation
4. Train/test split
5. Machine learning preprocessing pipeline
6. Logistic Regression model training
7. Model evaluation
8. Conversion probability prediction
9. Exporting prediction results
10. Save trained model using Joblib
11. Load trained model within the same notebook
12. Load future customer dataset
13. Score future customer datasets using the saved model
14. Export future customer prediction outputs

---

# Input Features

The model uses several customer behavioural and financial indicators including:

- Product Type
- Fresco Segment
- Age Band
- Region
- Marketing Channel
- Days to Maturity
- Account Value
- Monthly Contribution
- Active Direct Debit
- Failed Direct Debit
- Online Login Activity
- Complaints
- Email Open Rate
- Website Visits
- Previous Campaign Response
- Campaign Cost

---

## Preprocessing Pipeline

The model uses Scikit-learn Pipeline and ColumnTransformer for automated preprocessing:

- Numeric features scaled using StandardScaler
- Categorical features encoded using OneHotEncoder
- Logistic Regression used for conversion prediction

The pipeline automates preprocessing and prediction workflows in a reusable and production-style architecture.

---

# Model Outputs

The model generates:

- Predicted Conversion (0/1)
- Conversion Probability Score
- Customer Conversion Priority

Priority bands include:

- High Conversion Potential
- Medium Conversion Potential
- Low Conversion Potential
  
[Conversion potential represents the likelihood of a customer positively responding to outbound financial services campaigns through reinvestment, retention, product continuation, transfer, or upgrade activities.]

---

# Future Customer Prediction

The trained Logistic Regression model is saved using Joblib and reloaded within the same notebook to simulate a production-style machine learning workflow.

A separate future customer dataset containing 1,000 unseen customer records is then scored using the saved model to predict:

- Conversion likelihood
- Conversion probability
- Customer priority bands

This simulates a real-world outbound campaign scoring and customer propensity modelling process used in financial services and marketing analytics environments.

---

# Project Structure

```text
Campaign-Conversion-Prediction-Model/
│
├── data/
├── model/
├── notebooks/
├── output/
├── scripts/
└── README.md
```

---

#  Example Business Use Cases

* Outbound marketing prioritisation
*  Campaign conversion prediction
*  Customer propensity modelling
*  Marketing ROI optimisation
*  Next-best-action targeting
*  Financial services customer analytics

---

## Business Value

This project demonstrates how machine learning and behavioural analytics can be applied to improve campaign targeting, customer prioritisation, and marketing conversion performance.

The solution supports data-driven outbound marketing strategies by identifying customers with the highest likelihood of conversion, helping organisations optimise campaign efficiency and improve ROI.

---

# Author
Satheesh Gurusamy
