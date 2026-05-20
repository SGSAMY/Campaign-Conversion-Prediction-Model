#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

df = pd.read_excel(
    r"C:\& Satheesh\Campaign-Conversion-Prediction-Model\data\Financial_Services_KPI_Dashboard_5000Rows.xlsx"
)

df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

df.head()


# In[3]:


df["conversion_flag"] = (
    (
        (df["email_open_rate"] >= 40) &
        (df["web_visits_30d"] >= 5) &
        (df["previous_campaign_response"] == "Yes")
    )
    |
    (
        (df["active_dd"] == "Yes") &
        (df["failed_dd"] == "No") &
        (df["account_value"] >= 15000)
    )
).astype(int)

df["conversion_flag"].value_counts()


# In[5]:


features = [
    "product_type",
    "fresco_segment",
    "age_band",
    "region",
    "marketing_channel",
    "days_to_maturity",
    "account_value",
    "monthly_contribution",
    "active_dd",
    "failed_dd",
    "online_login_days",
    "complaints",
    "email_open_rate",
    "web_visits_30d",
    "previous_campaign_response",
    "campaign_cost"
]

target = "conversion_flag"

X = df[features]
y = df[target]


# In[7]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# In[9]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

categorical_features = [
    "product_type",
    "fresco_segment",
    "age_band",
    "region",
    "marketing_channel",
    "active_dd",
    "failed_dd",
    "previous_campaign_response"
]

numeric_features = [
    "days_to_maturity",
    "account_value",
    "monthly_contribution",
    "online_login_days",
    "complaints",
    "email_open_rate",
    "web_visits_30d",
    "campaign_cost"
]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=10000))
    ]
)

model.fit(X_train, y_train)


# In[11]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))


# In[13]:


y_probability = model.predict_proba(X_test)[:, 1]

results = X_test.copy()
results["actual_conversion"] = y_test.values
results["predicted_conversion"] = y_pred
results["conversion_probability"] = y_probability

results.head()


# In[15]:


def conversion_priority(probability):
    if probability >= 0.70:
        return "High Conversion Potential"
    elif probability >= 0.40:
        return "Medium Conversion Potential"
    else:
        return "Low Conversion Potential"

results["conversion_priority"] = results["conversion_probability"].apply(conversion_priority)

results.head()


# In[17]:


output_path = r"C:\& Satheesh\Campaign-Conversion-Prediction-Model\output\campaign_conversion_predictions.xlsx"

results.to_excel(output_path, index=False)

print("Prediction output exported successfully.")


# ## Saving Trained Model 

# In[21]:


import joblib

# Save trained model
joblib.dump(model, "conversion_model.pkl")

print("Model saved successfully.")


# ## Loading Trained Model

# In[23]:


import joblib

# Load trained model
loaded_model = joblib.load("conversion_model.pkl")

print("Model loaded successfully.")


# ## Reading Future Customer Dataset

# In[25]:


future_customers = pd.read_excel(
    r"C:\& Satheesh\Campaign-Conversion-Prediction-Model\data\future_customers_1000.xlsx"
)

future_customers.columns = future_customers.columns.str.strip().str.lower().str.replace(" ", "_")

future_customers.head()


# In[27]:


future_X = future_customers[features]


# In[29]:


loaded_model


# In[31]:


future_predictions = loaded_model.predict(future_X)

future_probabilities = loaded_model.predict_proba(future_X)[:, 1]


# In[33]:


future_results = future_customers.copy()

future_results["predicted_conversion"] = future_predictions

future_results["conversion_probability"] = future_probabilities

future_results.head()


# In[35]:


future_results["conversion_priority"] = future_results[
    "conversion_probability"
].apply(conversion_priority)

future_results.head()


# In[37]:


future_output_path = r"C:\& Satheesh\Campaign-Conversion-Prediction-Model\output\future_customer_predictions.xlsx"

future_results.to_excel(future_output_path, index=False)

print("Future customer predictions exported successfully.")


# In[ ]:




