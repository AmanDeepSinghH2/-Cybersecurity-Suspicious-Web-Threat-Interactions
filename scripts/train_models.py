"""Train Logistic Regression and KNN models on the intrusion dataset.

This script:
- Loads the CSV at ../Sheets/cybersecurity_intrusion_data.csv
- Applies a simple preprocessing pipeline (numeric scaling + one-hot for categoricals)
- Trains a Logistic Regression and a K-Nearest Neighbors classifier
- Evaluates on a holdout split and prints metrics
- Saves trained models to ./models/

Run:
    python scripts\train_models.py
"""
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "Sheets" / "cybersecurity_intrusion_data.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def build_preprocessor(numeric_features, categorical_features):
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    # sklearn >=1.2 uses `sparse_output` instead of `sparse`.
    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor


def evaluate_and_print(model_name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(f"\n=== {model_name} ===")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    # try ROC AUC if predict_proba exists
    if hasattr(model, "predict_proba"):
        try:
            y_score = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_score)
            print(f"ROC AUC: {auc:.4f}")
        except Exception:
            pass


def main():
    print(f"Loading data from: {DATA_PATH}")
    df = load_data(DATA_PATH)
    print(f"Data shape: {df.shape}")

    # Target and features (based on repo idea.txt and CSV header)
    target = "attack_detected"

    # Define sensible default features (numeric and categorical)
    numeric_features = [
        "network_packet_size",
        "login_attempts",
        "session_duration",
        "ip_reputation_score",
        "failed_logins",
        "unusual_time_access",
    ]

    categorical_features = [
        "protocol_type",
        "encryption_used",
        "browser_type",
    ]

    # If any feature is missing, automatically adjust
    available_cols = set(df.columns)
    numeric_features = [c for c in numeric_features if c in available_cols]
    categorical_features = [c for c in categorical_features if c in available_cols]

    if target not in available_cols:
        raise RuntimeError(f"Target column '{target}' not found in data. Columns: {available_cols}")

    X = df[numeric_features + categorical_features]
    y = df[target].astype(int)

    print("Class distribution:\n", y.value_counts(normalize=True))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    # Logistic Regression
    log_clf = Pipeline(steps=[("preproc", preprocessor), ("clf", LogisticRegression(max_iter=1000))])
    log_clf.fit(X_train, y_train)
    joblib.dump(log_clf, MODEL_DIR / "logistic_model.joblib")
    evaluate_and_print("Logistic Regression", log_clf, X_test, y_test)

    # K-Nearest Neighbors
    knn_clf = Pipeline(steps=[("preproc", preprocessor), ("clf", KNeighborsClassifier())])
    knn_clf.fit(X_train, y_train)
    joblib.dump(knn_clf, MODEL_DIR / "knn_model.joblib")
    evaluate_and_print("K-Nearest Neighbors", knn_clf, X_test, y_test)

    print(f"Models saved to: {MODEL_DIR}")


if __name__ == "__main__":
    main()
