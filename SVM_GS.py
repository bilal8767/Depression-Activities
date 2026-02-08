# =========================================================
# SVM GRID SEARCH + RANDOM SPLIT (80/20)
# =========================================================

import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.svm import SVC


# =========================
# CONFIG
# =========================
BASE_DIR = Path(__file__).resolve().parent
ZIP_PATH = BASE_DIR / "Feature Extraction OutputGS.zip"
EXTRACT_DIR = BASE_DIR / "feature_extracted_random_split"

TARGET_COL = "activity_label"
DROP_COLS = ["participant_id"]

TEST_SIZE = 0.2
RANDOM_STATE = 42
# =========================


def extract_zip(zip_path, extract_dir):
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)


def load_all_csvs(base_dir):
    csv_files = list(base_dir.rglob("*.csv"))
    if not csv_files:
        raise RuntimeError("No CSV files found")

    dfs = []
    for fp in csv_files:
        df = pd.read_csv(fp)
        if TARGET_COL not in df.columns:
            raise ValueError(f"Missing {TARGET_COL} in {fp}")
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)

    data[TARGET_COL] = (
        data[TARGET_COL].astype(str)
        .str.strip()
        .str.lower()
        .replace({"walkings": "walking"})
    )
    return data


def main():
    print("Extracting dataset...")
    extract_zip(ZIP_PATH, EXTRACT_DIR)

    print("Loading CSV files...")
    data = load_all_csvs(EXTRACT_DIR)
    print("Total samples:", data.shape[0])

    drop_cols = [TARGET_COL] + [c for c in DROP_COLS if c in data.columns]
    feature_cols = [c for c in data.columns if c not in drop_cols]

    X = data[feature_cols].to_numpy(dtype=np.float32)
    y = data[TARGET_COL].to_numpy()

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    class_names = list(le.classes_)

    print("Classes:", class_names)

    # =========================
    # RANDOM SPLIT (80/20)
    # =========================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc,
        test_size=TEST_SIZE,
        stratify=y_enc,
        random_state=RANDOM_STATE
    )

    print("Train size:", X_train.shape[0])
    print("Test size :", X_test.shape[0])

    # =========================
    # SVM + GRID SEARCH
    # =========================
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf"))
    ])

    param_grid = {
        "clf__C": [0.1, 1, 5, 10, 50],
        "clf__gamma": ["scale", 0.01, 0.1, 1]
    }

    print("\nRunning Grid Search on SVM...")
    grid = GridSearchCV(
        pipeline,
        param_grid,
        scoring="f1_macro",
        cv=5,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    print("\nBest parameters found:")
    print(grid.best_params_)

    best_model = grid.best_estimator_

    # =========================
    # EVALUATION
    # =========================
    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\nFinal Test Accuracy: {acc:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=class_names,
        digits=4
    ))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, xticks_rotation=45)
    ax.set_title("Confusion Matrix - SVM (Grid Search + Random Split)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
