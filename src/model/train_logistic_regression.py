import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


# --------------------------------------------------
# 1. Load processed dataset
# --------------------------------------------------

DATA_PATH = "data/processed/loans_clean_v1.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# --------------------------------------------------
# 2. Separate target and features
# --------------------------------------------------

target = "default_flag"

X = df.drop(columns=[target])
y = df[target]


# --------------------------------------------------
# 3. Handle date columns
# --------------------------------------------------

date_columns = ["issue_d", "earliest_cr_line"]

for col in date_columns:
    X[col] = pd.to_datetime(X[col], errors="coerce")

# Convert dates into useful numerical values
X["issue_year"] = X["issue_d"].dt.year
X["issue_month"] = X["issue_d"].dt.month

X["earliest_cr_year"] = X["earliest_cr_line"].dt.year
X["earliest_cr_month"] = X["earliest_cr_line"].dt.month

# Remove original string/date columns
X = X.drop(columns=date_columns)


# --------------------------------------------------
# 4. Handle missing values
# --------------------------------------------------

X = X.fillna(X.median(numeric_only=True))

# Convert boolean columns to integers
bool_columns = X.select_dtypes(include=["bool"]).columns

X[bool_columns] = X[bool_columns].astype(int)


# --------------------------------------------------
# 5. Train/Test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 6. Feature scaling
# --------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --------------------------------------------------
# 7. Logistic Regression
# --------------------------------------------------

model = LogisticRegression(
    class_weight="balanced",
    max_iter=2000,
    random_state=42
)

model.fit(X_train_scaled, y_train)


# --------------------------------------------------
# 8. Predictions
# --------------------------------------------------

y_pred = model.predict(X_test_scaled)

y_prob = model.predict_proba(X_test_scaled)[:, 1]


# --------------------------------------------------
# 9. Evaluation
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)


print("\n" + "=" * 60)
print("LOGISTIC REGRESSION RESULTS")
print("=" * 60)

print(f"Accuracy   : {accuracy:.4f}")
print(f"Precision  : {precision:.4f}")
print(f"Recall     : {recall:.4f}")
print(f"F1 Score   : {f1:.4f}")
print(f"ROC-AUC    : {roc_auc:.4f}")


print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Non-default", "Default"]
    )
)


import os
import joblib

os.makedirs("models", exist_ok=True)

joblib.dump(
    {
        "model": model,
        "scaler": scaler
    },
    "models/logistic_regression.pkl"
)

print("\nModel saved to models/logistic_regression.pkl")