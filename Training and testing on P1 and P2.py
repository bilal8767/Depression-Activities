 

import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier


# =========================
# CONFIG
# =========================
BASE_DIR = Path(__file__).resolve().parent
ZIP_PATH = BASE_DIR / "Training and Testing.zip"
EXTRACT_DIR = BASE_DIR / "unzipped_Training_and_Testing"

# Use P_1 and P_2 for testing, P_3, P_4, P_5, P_6 for training
TEST_FOLDERS = {"P_1", "P_2"}
TARGET_COL = "activity_label"
DROP_COLS = ["participant_id"]

OUTPUT_DIR = BASE_DIR / "p1_p2_test_outputs"
CM_DIR = OUTPUT_DIR / "confusion_matrices"

# Always use 'Training and Testing Dataset' subfolder if it exists
def find_base_dir(extract_dir: Path) -> Path:
    candidate = extract_dir / 'Training and Testing Dataset'
    if candidate.exists() and candidate.is_dir():
        return candidate
    return extract_dir
# =========================


def extract_zip(zip_path: Path, extract_dir: Path) -> None:
    """Extract a zip file to `extract_dir`.

    `zip_path` is expected to be a Path to an existing zip file.
    """
    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(zip_path), "r") as z:
        z.extractall(extract_dir)


def resolve_zip_path(zip_path_str: str) -> Path:
    """Resolve the zip path in a platform-robust way.

    Tries, in order:
      - the provided path as-is
      - the same filename next to this script
      - a glob search for matching zip names in the script dir, its parents, and cwd
    Raises FileNotFoundError with attempted locations if not found.
    """
    target = Path(zip_path_str)
    # 1) as given
    if target.exists():
        return target

    tried = [str(target)]

    # 2) next to this script
    script_dir = Path(__file__).resolve().parent
    candidate = script_dir / target.name
    tried.append(str(candidate))
    if candidate.exists():
        return candidate

    # 3) glob search for similar names
    stem = target.stem
    for d in [script_dir] + list(script_dir.parents) + [Path.cwd()]:
        for match in d.glob(f"*{stem}*.zip"):
            return match

    raise FileNotFoundError(
        f"Zip file not found. Attempted locations: {tried} and searched script dir, parents, and cwd for '*{stem}*.zip'"
    )





def resolve_folder_case_insensitive(base_dir: Path, name: str) -> Path:
    # exact
    p = base_dir / name
    if p.exists() and p.is_dir():
        return p
    # case-insensitive
    for d in base_dir.iterdir():
        if d.is_dir() and d.name.lower() == name.lower():
            return d
    raise FileNotFoundError(f"Folder '{name}' not found under {base_dir}.")


def load_csvs_in_folder(folder: Path) -> pd.DataFrame:
    csvs = sorted(folder.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in: {folder}")
    dfs = []
    for fp in csvs:
        df = pd.read_csv(fp)
        if TARGET_COL not in df.columns:
            raise ValueError(f"Missing '{TARGET_COL}' in {fp}. Columns: {list(df.columns)}")
        dfs.append(df)
    result = pd.concat(dfs, ignore_index=True)
    # fix typo: map 'walkings' -> 'walking' (occurs in Samyak/walking.csv)
    if TARGET_COL in result.columns:
        result[TARGET_COL] = result[TARGET_COL].astype(str).str.strip().str.lower()
        result[TARGET_COL] = result[TARGET_COL].replace({'walkings': 'walking'})
    return result


def save_confusion_matrix_png(cm, class_names, out_path: Path, title: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, xticks_rotation=45, values_format="d")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def evaluate(name, model, X_train, y_train, X_test, y_test, class_names, test_label):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec_macro  = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro   = f1_score(y_test, y_pred, average="macro", zero_division=0)

    prec_w = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec_w  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_w   = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # ensure reports/matrices include all classes seen during training
    full_labels = list(range(len(class_names)))
    cm = confusion_matrix(y_test, y_pred, labels=full_labels)

    print("\n" + "=" * 95)
    print(f"MODEL: {name} | TEST={test_label}")
    print(f"Accuracy            : {acc:.4f}")
    print(f"Precision (macro)   : {prec_macro:.4f} | Precision (weighted): {prec_w:.4f}")
    print(f"Recall (macro)      : {rec_macro:.4f} | Recall (weighted)   : {rec_w:.4f}")
    print(f"F1 (macro)          : {f1_macro:.4f} | F1 (weighted)       : {f1_w:.4f}")
    print("-" * 95)
    print("Classification report:")
    print(classification_report(y_test, y_pred, labels=full_labels, target_names=class_names, digits=4, zero_division=0))
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)

    return cm


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CM_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Resolve & extract
    zip_file = resolve_zip_path(ZIP_PATH)
    print(f"Using zip file: {zip_file}")
    extract_zip(zip_file, EXTRACT_DIR)

    # 2) Find base dir with participant folders
    base_dir = find_base_dir(EXTRACT_DIR)
    all_folders = sorted([p for p in base_dir.iterdir() if p.is_dir()])
    if not all_folders:
        raise FileNotFoundError(f"No participant folders found under: {base_dir}")

    # 3) Manually set test and train folders to match desired split
    test_dirs = [resolve_folder_case_insensitive(base_dir, "P_3"), resolve_folder_case_insensitive(base_dir, "P_1")]
    train_dirs = [resolve_folder_case_insensitive(base_dir, "P_2"),
                  resolve_folder_case_insensitive(base_dir, "P_4"),
                  resolve_folder_case_insensitive(base_dir, "P_5"),
                  resolve_folder_case_insensitive(base_dir, "P_6")]

    print("Train folders:", [p.name for p in train_dirs])
    print("Test folders :", [p.name for p in test_dirs])

    # 4) Load data
    train_df = pd.concat([load_csvs_in_folder(p) for p in train_dirs], ignore_index=True)
    test_df = pd.concat([load_csvs_in_folder(p) for p in test_dirs], ignore_index=True)

    print("\nTrain shape:", train_df.shape)
    print("Test shape :", test_df.shape)

    # 5) Features/labels
    # normalize labels (lowercase/strip)
    train_df[TARGET_COL] = train_df[TARGET_COL].astype(str).str.strip().str.lower()
    test_df[TARGET_COL] = test_df[TARGET_COL].astype(str).str.strip().str.lower()

    # compute label counts across train+test and keep only present labels
    combined = pd.concat([train_df[[TARGET_COL]], test_df[[TARGET_COL]]], ignore_index=True)
    counts = combined[TARGET_COL].value_counts()
    present_labels = sorted([lab for lab, cnt in counts.items() if cnt > 0])

    if not present_labels:
        raise ValueError("No labels found in train/test datasets.")

    # remove any classes that are clearly absent (e.g., accidental 'walkings')
    print("Labels present in data:", present_labels)

    drop_cols = [TARGET_COL] + [c for c in DROP_COLS if c in train_df.columns]
    feature_cols = [c for c in train_df.columns if c not in drop_cols]

    missing = set(feature_cols) - set(test_df.columns)
    if missing:
        raise ValueError(f"Test set missing feature columns: {sorted(missing)}")

    X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    X_test  = test_df[feature_cols].to_numpy(dtype=np.float32)

    # Fit encoder only on present labels so phantom classes (support=0) are excluded
    le = LabelEncoder()
    le.fit(present_labels)
    y_train = le.transform(train_df[TARGET_COL].astype(str).values)
    y_test  = le.transform(test_df[TARGET_COL].astype(str).values)
    class_names = list(le.classes_)

    # 6) Models (defined after class_names so XGBoost can get correct num_class)
    models = {
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=7))
        ]),
        "SVM_RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=5.0, gamma="scale"))
        ]),
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, class_weight="balanced"))
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample"
        ),
        "XGBoost": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators=600,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="multi:softprob",
                eval_metric="mlogloss",
                random_state=42,
                n_jobs=-1,
                num_class=len(class_names)
            ))
        ]),
    }

    test_label = " + ".join([d.name for d in test_dirs])

    # 7) Train/Evaluate + save confusion matrices
    for name, model in models.items():
        cm = evaluate(name, model, X_train, y_train, X_test, y_test, class_names, test_label)

        cm_path = CM_DIR / f"confusion_matrix_{name}_test_{test_label.replace(' ', '').replace('+','_')}.png"
        save_confusion_matrix_png(
            cm, class_names, cm_path,
            title=f"Confusion Matrix - {name} (Test={test_label})"
        )
        print("Saved confusion matrix:", cm_path)


if __name__ == "__main__":
    main()