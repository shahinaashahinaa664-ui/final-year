
# 1️⃣ DATA HANDLING
# ==============================
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# # Data preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# # Modeling (Classification for Loan Dataset)
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report


# ==============================
# 2️⃣ LOAD DATA
# ==============================
df = pd.read_excel(r"C:\Users\Shahina\Downloads\LOAN.xlsx")

print("First 5 Rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())


# ==============================
# 3️⃣ DATA CLEANING
# ==============================

# Remove duplicates
df = df.drop_duplicates()

# Remove datetime columns (VERY IMPORTANT FIX)
for col in df.select_dtypes(include=['datetime64[ns]']).columns:
    df.drop(col, axis=1, inplace=True)

# Fill numeric missing values with mean
numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Fill categorical missing values with mode
categorical_cols = df.select_dtypes(include='object').columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())


# ==============================
# 4️⃣ ENCODE CATEGORICAL DATA
# ==============================
le = LabelEncoder()

for col in categorical_cols:
    if col not in ["membership_category", "region_category"]:  
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])


# ==============================
# 5️⃣ DATA MODELING
# ==============================

# Assume last column is target (Loan_Status)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Ensure only numeric data in features
X = X.select_dtypes(include=[np.number])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


## ==============================
# 📊 COMPLETE PROFESSIONAL VISUALIZATION
# ==============================
# ==========================================
# 🎓 PROFESSIONAL ACADEMIC BACKGROUND STYLE
# ==========================================

import matplotlib.pyplot as plt
import seaborn as sns

# Clean academic theme
sns.set_theme(style="whitegrid")

# Soft outer background (entire figure)
plt.rcParams["figure.facecolor"] = "#eef2f7"   # light blue-grey

# White plotting area (inside chart)
plt.rcParams["axes.facecolor"] = "#ffffff"

# Borders
plt.rcParams["axes.edgecolor"] = "#444444"
plt.rcParams["axes.linewidth"] = 1.2

# Grid style
plt.rcParams["grid.color"] = "#d9d9d9"
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.alpha"] = 0.6

# Fonts
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelsize"] = 12

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (8,5)

# ---------------------------------
# 1️⃣ Membership Category vs Avg Login Frequency
# ---------------------------------
if "membership_category" in df.columns and "avg_frequency_login_days" in df.columns:

    avg_data = df.groupby("membership_category")["avg_frequency_login_days"].mean().reset_index()

    plt.figure(figsize=(10,6))   # Bigger figure for better spacing

    ax = sns.barplot(
        x="membership_category",
        y="avg_frequency_login_days",
        data=avg_data,
       palette="Blues"
    )

    # Title
    plt.title("Average Login Frequency by Membership Category",
              fontsize=14, fontweight="bold")

    # Axis labels
    plt.xlabel("Membership Category",
               fontsize=12, fontweight="bold", labelpad=15)

    plt.ylabel("Average Frequency Login Days",
               fontsize=12, fontweight="bold")

    # Rotate labels properly
    plt.xticks(rotation=25, ha="right", fontsize=10)

    # Add value labels on top of bars (Professional touch)
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.0f}",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom',
                    fontsize=9,
                    xytext=(0, 5),
                    textcoords='offset points')

    plt.tight_layout()
    plt.show()

else:
    print("Required columns not found.")



# Donut Chart – Sum of Churn Risk by Region
# ---------------------------------

if "churn_risk_score" in df.columns and "region_category" in df.columns:

    churn_sum = df.groupby("region_category")["churn_risk_score"].sum().reset_index()

    values = churn_sum["churn_risk_score"]
    labels = churn_sum["region_category"]

    plt.figure(figsize=(8,6))
    plt.gca().set_facecolor("#ffffff")   # keeps donut center clean

    wedges, texts, autotexts = plt.pie(
        values,
        labels=None,
        autopct=lambda pct: f"{int(round(pct/100.*sum(values)))}\n({pct:.2f}%)",
        startangle=90,
        wedgeprops={'width':0.4}
    )

    plt.title("Sum of churn_risk_score by region_category", fontsize=13)

    plt.legend(
        wedges,
        labels,
        title="region_category",
        loc="center left",
        bbox_to_anchor=(1, 0.5)
    )

    plt.tight_layout()
    plt.show()
# ---------------------------------
# Train Model
# ---------------------------------
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


# ---------------------------------
# 3️⃣ Feature Importance (Lollipop Style)
# ---------------------------------
importance = model.feature_importances_

plt.figure()
plt.stem(X.columns, importance)
plt.xticks(rotation=45)
plt.title("Feature Importance")
plt.tight_layout()
plt.show()


# ---------------------------------
# 4️⃣ Confusion Matrix
# ---------------------------------
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Rejected (0)", "Approved (1)"],
            yticklabels=["Rejected (0)", "Approved (1)"])

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()


# ---------------------------------
# 5️⃣ Gender vs Churn Risk
# ---------------------------------
if "gender" in df.columns and "churn_risk_score" in df.columns:
    plt.figure()
    sns.countplot(x="gender", hue="churn_risk_score", data=df)
    plt.title("Gender vs Churn Risk")
    plt.xlabel("Gender")
    plt.ylabel("Number of Customers")
    plt.tight_layout()
    plt.show()


# ---------------------------------
# 6️⃣ Age Distribution by Churn Risk
# ---------------------------------
if "age" in df.columns and "churn_risk_score" in df.columns:
    plt.figure()

    sns.boxplot(
        x="churn_risk_score",
        y="age",
        data=df,
        palette=["#4C758F", "#1E297D"]   # 0 = Green, 1 = Red
    )

    plt.title("Age Distribution by Churn Risk", fontsize=14, fontweight="bold")
    plt.xlabel("Churn Risk (0 = No, 1 = Yes)", fontsize=12)
    plt.ylabel("Age", fontsize=12)
    plt.tight_layout()
    plt.show()

print("\n✅ Complete Professional Visualizations Generated Successfully.")
