import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
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


# ============================================================
# 1. Load processed dataset
# ============================================================

DATA_PATH = "data/processed/loans_clean_v1.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# ============================================================
# 2. Separate features and target
# ============================================================

TARGET = "default_flag"

X = df.drop(columns=[TARGET])
y = df[TARGET]


# ============================================================
# 3. Convert date columns
# ============================================================

date_columns = ["issue_d", "earliest_cr_line"]

for col in date_columns:
    X[col] = pd.to_datetime(X[col], errors="coerce")


# Extract useful information from dates

X["issue_year"] = X["issue_d"].dt.year
X["issue_month"] = X["issue_d"].dt.month

X["earliest_cr_year"] = X["earliest_cr_line"].dt.year
X["earliest_cr_month"] = X["earliest_cr_line"].dt.month


# Remove original date columns

X = X.drop(columns=date_columns)


# ============================================================
# 4. Convert Boolean columns to integers
# ============================================================

bool_columns = X.select_dtypes(include=["bool"]).columns

X[bool_columns] = X[bool_columns].astype(int)


# ============================================================
# 5. Train/Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training rows:", len(X_train))
print("Testing rows :", len(X_test))


# ============================================================
# 6. Handle missing values
#    Calculate medians ONLY from training data
# ============================================================

train_medians = X_train.median(numeric_only=True)

X_train = X_train.fillna(train_medians)
X_test = X_test.fillna(train_medians)


# ============================================================
# 7. Feature Scaling
#    Fit scaler ONLY on training data
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# 8. Logistic Regression Hyperparameter Tuning
# ============================================================

print("\nTuning Logistic Regression...")

logistic = LogisticRegression(
    max_iter=2000,
    random_state=42
)


param_grid = {
    "C": [0.01, 0.1, 1, 10, 100],
    "solver": ["liblinear", "lbfgs"],
    "class_weight": ["balanced", None]
}


search = RandomizedSearchCV(
    estimator=logistic,
    param_distributions=param_grid,
    n_iter=10,
    scoring="roc_auc",
    cv=3,
    random_state=42,
    n_jobs=-1,
    verbose=1
)


search.fit(X_train_scaled, y_train)


# Best model

model = search.best_estimator_


print("\nLogistic Regression tuning completed.")

print("Best parameters:")
print(search.best_params_)

print("\nBest cross-validation ROC-AUC:")
print(f"{search.best_score_:.4f}")


# ============================================================
# 9. Predictions
# ============================================================

y_prob = model.predict_proba(X_test_scaled)[:, 1]


# Default classification threshold

threshold = 0.50

y_pred = (y_prob >= threshold).astype(int)


# ============================================================
# 10. Calculate Metrics
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_prob
)


# ============================================================
# 11. Display Results
# ============================================================

print("\n" + "=" * 60)
print(" LOGISTIC REGRESSION RESULTS")
print("=" * 60)

print(f"Accuracy   : {accuracy:.4f}")
print(f"Precision  : {precision:.4f}")
print(f"Recall     : {recall:.4f}")
print(f"F1 Score   : {f1:.4f}")
print(f"ROC-AUC    : {roc_auc:.4f}")


# ============================================================
# 12. Confusion Matrix
# ============================================================

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ============================================================
# 13. Classification Report
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Non-default", "Default"],
        zero_division=0
    )
)


# ============================================================
# 14. Save Model
# ============================================================

os.makedirs("models", exist_ok=True)


model_data = {
    "model": model,
    "scaler": scaler,
    "feature_columns": X_train.columns.tolist(),
    "train_medians": train_medians.to_dict(),
    "threshold": threshold
}


joblib.dump(
    model_data,
    "models/logistic_regression.pkl"
)


print("\nModel saved to models/logistic_regression.pkl")


# ============================================================
# 15. Save Metrics
# ============================================================

os.makedirs("reports", exist_ok=True)


metrics = pd.DataFrame({
    "Model": ["Logistic Regression"],
    "Accuracy": [accuracy],
    "Precision": [precision],
    "Recall": [recall],
    "F1": [f1],
    "ROC-AUC": [roc_auc]
})


metrics.to_csv(
    "reports/logistic_regression_metrics.csv",
    index=False
)


print("Metrics saved to reports/logistic_regression_metrics.csv")