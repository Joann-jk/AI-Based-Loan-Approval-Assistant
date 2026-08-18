import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

DATA_PATH = "data/processed/loans_clean_v1.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("Columns:")
print(df.columns.tolist())

# Convert date columns into useful numerical features
df["issue_d"] = pd.to_datetime(df["issue_d"], errors="coerce")

df["issue_year"] = df["issue_d"].dt.year
df["issue_month"] = df["issue_d"].dt.month

# Raw date columns are not used directly by Random Forest
df = df.drop(columns=["issue_d", "earliest_cr_line"])

# Target variable
y = df["default_flag"]

# Features
X = df.drop(columns=["default_flag"])

# Convert all feature columns to numeric values
X = X.astype(float)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("X data types:")
print(X.dtypes)

print("Non-numeric columns:")
print(X.select_dtypes(exclude="number").columns.tolist())

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)
print("Training target:", y_train.shape)
print("Testing target:", y_test.shape)

# Random Forest hyperparameter tuning
print("\nTuning Random Forest...")

rf = RandomForestClassifier(
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

param_grid = {
    "n_estimators": [200, 300],
    "max_depth": [10, 15, 20, None],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"]
}

search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_grid,
    n_iter=6,
    scoring="roc_auc",
    cv=3,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

search.fit(X_train, y_train)

model = search.best_estimator_

print("\nRandom Forest tuning completed.")
print("Best parameters:")
print(search.best_params_)
print("Best cross-validation ROC-AUC:", search.best_score_)

# Probability of default
y_prob = model.predict_proba(X_test)[:, 1]



#Change the decision threshold
threshold = 0.45
y_pred = (y_prob >= threshold).astype(int)

print("Predictions completed.")

# Feature importance
importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    by="importance",
    ascending=False
)

print("\nTop 15 Important Features:")
print(importance.head(15))

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ROC-AUC
roc_auc = roc_auc_score(y_test, y_prob)

print("\nROC-AUC Score:", roc_auc)