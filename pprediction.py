# ================================
# STEP 1: Import Libraries
# ================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import matplotlib.pyplot as plt


# ================================
# STEP 2: Load Dataset
# ================================

df = pd.read_csv(r"C:\Users\Shahina\Downloads\loan_prediction_dataset.csv")

print("First 5 rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())


# ================================
# STEP 3: Remove Duplicates
# ================================

df = df.drop_duplicates()


# ================================
# STEP 4: Handle Missing Values
# ================================

# Numerical columns
num_cols = df.select_dtypes(include=np.number).columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# Categorical columns
# use 'object' instead of 'str' for categorical columns
cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])


# ================================
# STEP 5: Drop High Cardinality Columns
# ================================

# Drop columns that have too many unique values
for col in df.columns:
    if df[col].nunique() > 50:
        print("Dropping high-cardinality column:", col)
        df = df.drop(col, axis=1)


print("\n==============================")
print("MODEL 1: MEMBERSHIP CATEGORY PREDICTION")
print("==============================")

# Define target
X1 = df.drop('membership_category', axis=1)
y1 = df['membership_category']

# Encode target
le = LabelEncoder()
y1 = le.fit_transform(y1)

# Encode features
X1 = pd.get_dummies(X1, drop_first=True)

# Train-Test Split
X1_train, X1_test, y1_train, y1_test = train_test_split(
    X1, y1, test_size=0.2, random_state=42
)

# Train Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X1_train, y1_train)

# Predictions
y1_pred = clf.predict(X1_test)

# Accuracy
acc = accuracy_score(y1_test, y1_pred)
print("Classification Accuracy:", round(acc*100,2), "%")
# ================================
# Trend Graph: Actual vs Predicted
# ================================

plt.figure(figsize=(10,5))

# Plot first 50 samples for clear visualization
# y1_test is numpy array after train_test_split, so use it directly
plt.plot(y1_test[:50], label="Actual", marker='o')
plt.plot(y1_pred[:50], label="Predicted", marker='x')

plt.title("RandomForestClassifier: Actual vs Predicted Membership Category",
          fontweight="bold")

plt.xlabel("Customer Index")
plt.ylabel("Membership Category (Encoded)")
plt.legend()
plt.grid(alpha=0.3)

plt.savefig("membership_prediction_plot.png")
plt.close()

print("\n==============================")
print("MODEL 2: CHURN RISK SCORE PREDICTION")
print("==============================")

# Define target
X2 = df.drop('churn_risk_score', axis=1)
y2 = df['churn_risk_score']

# Encode features
X2 = pd.get_dummies(X2, drop_first=True)

# Train-Test Split
X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.2, random_state=42
)

# Train Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(X2_train, y2_train)

# Predictions
y2_pred = reg.predict(X2_test)

# Evaluation
mae = mean_absolute_error(y2_test, y2_pred)
r2 = r2_score(y2_test, y2_pred)

print("Mean Absolute Error:", round(mae,2))
print("R2 Score:", round(r2,2))
# ================================
# MINI DASHBOARD - CHURN RISK
# ================================

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# Prepare data
results_df = pd.DataFrame({
    "Actual": y2_test.reset_index(drop=True),
    "Predicted": y2_pred
})

# Sort for smoother trend
results_df = results_df.sort_values(by="Actual").reset_index(drop=True)

# Calculate error
results_df["Error"] = results_df["Actual"] - results_df["Predicted"]

# Create subplots
fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=("Trend: Actual vs Predicted Churn Risk",
                    "Prediction Error Distribution"),
    vertical_spacing=0.15
)

# ---- Trend Graph ----
fig.add_trace(
    go.Scatter(
        y=results_df["Actual"][:100],
        mode='lines+markers',
        name="Actual"
    ),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(
        y=results_df["Predicted"][:100],
        mode='lines+markers',
        name="Predicted"
    ),
    row=1, col=1
)

# ---- Error Histogram ----
fig.add_trace(
    go.Histogram(
        x=results_df["Error"],
        name="Error Distribution"
    ),
    row=2, col=1
)

# Layout
fig.update_layout(
    height=700,
    title="Customer Churn Risk Prediction Dashboard (Random Forest)",
    showlegend=True
)

fig.update_xaxes(title_text="Customer Index (Sorted)", row=1, col=1)
fig.update_yaxes(title_text="Churn Risk Score", row=1, col=1)

fig.update_xaxes(title_text="Prediction Error", row=2, col=1)
fig.update_yaxes(title_text="Frequency", row=2, col=1)

fig.show()

# ================================
# Interactive Churn Risk Graph
# ================================

# ================================
# Interactive Trend Graph - Churn Risk
# ================================

import plotly.graph_objects as go
import pandas as pd

# Sort by actual values for smoother trend
sorted_indices = y2_test.argsort()

actual_sorted = y2_test.iloc[sorted_indices].reset_index(drop=True)
pred_sorted = pd.Series(y2_pred[sorted_indices]).reset_index(drop=True)

fig = go.Figure()

# Actual Line
fig.add_trace(go.Scatter(
    y=actual_sorted[:100],
    mode='lines+markers',
    name='Actual Churn Risk'
))

# Predicted Line
fig.add_trace(go.Scatter(
    y=pred_sorted[:100],
    mode='lines+markers',
    name='Predicted Churn Risk'
))

fig.update_layout(
    title="Random Forest Regression – Churn Risk Analysis: Actual vs Predicted Churn Risk",
    xaxis_title="Customer Index (Sorted)",
    yaxis_title="Churn Risk Score",
    hovermode="x unified"
)

fig.show()
