# ===================== IMPORTS =====================
import os
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier


# ===================== UNZIP DATASET =====================
ZIP_PATH = "Training dataset.zip"
DATA_PATH = "dataset"

with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall(DATA_PATH)

print("Dataset extracted successfully")


# ===================== LOAD CSV FILES =====================
csv_files = []
for root, dirs, files in os.walk(DATA_PATH):
    for file in files:
        if file.endswith(".csv"):
            csv_files.append(os.path.join(root, file))

print("Total CSV files:", len(csv_files))


# ===================== FEATURE EXTRACTION =====================
X, y = [], []

for file in csv_files:
    df = pd.read_csv(file)
    df = df.select_dtypes(include=np.number)

    if df.shape[1] == 0:
        continue

    features = []
    for col in df.columns:
        features.extend([
            df[col].mean(),
            df[col].std(),
            df[col].min(),
            df[col].max()
        ])

    X.append(features)
    y.append(os.path.basename(os.path.dirname(file)))

X = np.array(X)
y = np.array(y)

print("Feature shape:", X.shape)

# Handle missing feature values by imputing column means
nan_count = np.isnan(X).sum()
if nan_count:
    print(f"Found {nan_count} missing values in features — imputing with column means")
    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)


# ===================== LABEL ENCODING =====================
le = LabelEncoder()
y = le.fit_transform(y)


# ===================== TRAIN TEST SPLIT =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ===================== SCALING =====================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ===================== MODELS =====================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
    "SVM": SVC(kernel="rbf"),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss"
    )
}


# ===================== TRAIN, EVALUATE & CONFUSION MATRIX =====================
for name, model in models.items():
    print("\n===================================")
    print("Model:", name)

    if name in ["Logistic Regression", "KNN", "SVM"]:
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

    # Accuracy
    acc = accuracy_score(y_test, preds)
    print("Accuracy:", round(acc, 3))

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, preds, target_names=le.classes_))

    # Confusion Matrix
    cm = confusion_matrix(y_test, preds)

    plt.figure()
    plt.imshow(cm)
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(range(len(le.classes_)), le.classes_, rotation=45)
    plt.yticks(range(len(le.classes_)), le.classes_)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()
    plt.show()


print("\nAll models trained & evaluated successfully ✅")