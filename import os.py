import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import warnings
warnings.filterwarnings("ignore")
import joblib
import time
import argparse

 

def find_dataset_dir(root_dirs=None, target_name="A_DeviceMotion_data"):
    """Search for a directory named target_name inside the provided root_dirs.
    Returns the first matching absolute path that contains CSV files, or None.
    """
    if root_dirs is None:
        root_dirs = [os.path.dirname(__file__) if '__file__' in globals() else os.getcwd(), os.getcwd()]

    for start in root_dirs:
        for dirpath, dirnames, filenames in os.walk(start):
            
            if os.path.basename(dirpath) == target_name:
                 
                for _, _, files in os.walk(dirpath):
                    for f in files:
                        if f.endswith('.csv') and not f.startswith('._'):
                            return dirpath
    return None


 
parser = argparse.ArgumentParser(description='Train classifiers on A_DeviceMotion_data dataset')
parser.add_argument('--dataset', '-d', help='Path to base A_DeviceMotion_data folder')
parser.add_argument('--max-rows', type=int, default=None, help='If set, randomly sample this many rows for quick testing')
parser.add_argument('--sample-frac', type=float, default=None, help='If set, randomly sample this fraction of rows for quick testing')
parser.add_argument('--n-jobs', type=int, default=1, help='Number of parallel jobs for CV and estimators (default 1)')
parser.add_argument('--no-cv', action='store_true', help='Disable cross-validation (faster)')
parser.add_argument('--fast', action='store_true', help='Fast mode: fewer estimators and no CV')
parser.add_argument('--n-estimators', type=int, default=None, help='Override n_estimators for ensemble models')
args, unknown = parser.parse_known_args()

base_path = None
 
if len(sys.argv) > 1 and not args.dataset:
    candidate = sys.argv[1]
    if os.path.exists(candidate):
        base_path = candidate
    else:
        print(f"Warning: provided dataset path does not exist: {candidate}")

 
if base_path is None and args.dataset:
    if os.path.exists(args.dataset):
        base_path = args.dataset
    else:
        print(f"Warning: --dataset path does not exist: {args.dataset}")

 
if base_path is None and os.environ.get("DATASET_PATH"):
    candidate = os.environ.get("DATASET_PATH")
    if os.path.exists(candidate):
        base_path = candidate
    else:
        print(f"Warning: DATASET_PATH is set but path does not exist: {candidate}")

 
legacy_path = "/mnt/data/extracted_dataset/A_DeviceMotion_data/A_DeviceMotion_data"
if base_path is None and os.path.exists(legacy_path):
    base_path = legacy_path

 
if base_path is None:
    found = find_dataset_dir()
    if found:
        base_path = found

if base_path is None:
    print("Could not find 'A_DeviceMotion_data' directory automatically.")
    print("Please set `base_path` to the extracted dataset folder containing activity subfolders.")
    print("Searched from:", os.path.dirname(__file__) if '__file__' in globals() else os.getcwd())
    sys.exit(1)

 
dataframes = []
csv_paths = []
for root, dirs, files in os.walk(base_path):
    for file in files:
        if file.endswith(".csv") and not file.startswith("._"):
            label = os.path.basename(root)  # folder name as activity label
            file_path = os.path.join(root, file)
            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                print(f"Warning: failed to read {file_path}: {e}")
                continue
            df["activity"] = label
            dataframes.append(df)
            csv_paths.append(file_path)

if not dataframes:
    print(f"No CSV files were found under {base_path} (or all reads failed).")
    print("Make sure the folder contains CSV files. Example structure: <base_path>/<activity_label>/sub_1.csv")
    sys.exit(1)

 
try:
    data = pd.concat(dataframes, ignore_index=True)
except ValueError as e:
    if "No objects to concatenate" in str(e):
        print("ValueError: No objects to concatenate — the list 'dataframes' is empty.")
        print(f"base_path = {base_path}")
        print(f"Number of CSV paths discovered: {len(csv_paths)}")
        if csv_paths:
            print("Sample CSV paths:")
            for p in csv_paths[:10]:
                print("  ", p)
        else:
            print("No CSV files were found under the provided base_path.")
        print("Make sure the dataset is extracted and the path points to the folder containing activity subfolders.")
        sys.exit(1)
    else:
        raise
 
if "activity" not in data.columns:
    print("Error: 'activity' column not present after reading files.")
    sys.exit(1)

# Record activity categories and convert labels to integer codes (int32) to save memory
categories = data["activity"].astype('category').cat.categories
print("Combined dataset shape:", data.shape)
print("Activity labels:", list(categories))

# y as integer codes (int32) - smaller than int64
y = data["activity"].astype('category').cat.codes.astype('int32')
X = data.drop(columns=["activity"])

# Convert all feature columns to numeric where possible, non-convertible -> NaN
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

# Downcast numeric columns to float32 to reduce memory usage
num_cols = X.select_dtypes(include=[np.number]).columns
if len(num_cols) > 0:
    X[num_cols] = X[num_cols].astype(np.float32)

 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4, stratify=y)

 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

 
models = {
    "LogisticRegression": LogisticRegression(max_iter=2, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=2, random_state=42),
    "GaussianNB": GaussianNB(),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "ExtraTrees": ExtraTreesClassifier(n_estimators=2, random_state=42),
}

 
# Only the requested classifiers are used (LogisticRegression, RandomForest,
# GradientBoosting, KNeighbors, AdaBoost, GaussianNB). Optional boosted
# libraries (XGBoost/LightGBM) are intentionally not included to keep the
# experiment consistent and lightweight.

results = []
 
if args.fast:
    print("Fast mode enabled: reducing ensemble sizes and disabling CV")
    for k in list(models.keys()):
        if 'Forest' in k or k in ('ExtraTrees', 'AdaBoost', 'GradientBoosting', 'XGBoost', 'LightGBM'):
            try:
                models[k].set_params(n_estimators=50)
            except Exception:
                pass

if args.n_estimators is not None:
    for k in list(models.keys()):
        try:
            models[k].set_params(n_estimators=args.n_estimators)
        except Exception:
            pass

cv = None if args.no_cv or args.fast else StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print('\nTraining and evaluating models (CV + test set)...')
start_all = time.time()
for name, model in models.items():
    print(f"\n==> {name}")
  
    cv_scores = []
    if cv is not None:
        try:
           
            use_threading = (os.name == 'nt' and args.n_jobs and args.n_jobs != 1)
            if use_threading:
                with joblib.parallel_backend('threading'):
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='f1_weighted', n_jobs=args.n_jobs)
            else:
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='f1_weighted', n_jobs=args.n_jobs)
            print(f"  CV f1_weighted: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f} (n={len(cv_scores)})")
        except Exception as e:
            print(f"  CV failed for {name}: {e}")
            cv_scores = []

    t0 = time.time()
    try:
      
        try:
            model.set_params(n_jobs=args.n_jobs)
        except Exception:
            pass

 
        use_threading_fit = (os.name == 'nt' and args.n_jobs and args.n_jobs != 1)
        try:
            if use_threading_fit:
                with joblib.parallel_backend('threading'):
                    model.fit(X_train_scaled, y_train)
            else:
                model.fit(X_train_scaled, y_train)
        except ModuleNotFoundError as mnfe:
            if '_posixsubprocess' in str(mnfe):
                print(f"  Caught ModuleNotFoundError('_posixsubprocess') while training {name}.")
                print("  This can happen when process-based parallelism tries to import POSIX-only modules on Windows.")
                print("  Retrying training with single-threaded backend (n_jobs=1)...")
                try:
                    try:
                        model.set_params(n_jobs=1)
                    except Exception:
                        pass
                    model.fit(X_train_scaled, y_train)
                except Exception as e2:
                    print(f"  Retry failed for {name}: {e2}")
                    raise
            else:
                raise
    except Exception as e:
        print(f"  Failed to train {name}: {e}")
        continue
    t1 = time.time()
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average='weighted', zero_division=0)
    rec = recall_score(y_test, preds, average='weighted', zero_division=0)
    f1 = f1_score(y_test, preds, average='weighted', zero_division=0)
    results.append({
        'name': name,
        'model': model,
        'cv_mean_f1': float(cv_scores.mean()) if len(cv_scores) else None,
        'cv_std_f1': float(cv_scores.std()) if len(cv_scores) else None,
        'test_accuracy': float(acc),
        'test_precision': float(prec),
        'test_recall': float(rec),
        'test_f1': float(f1),
        'fit_time_s': float(t1 - t0)
    })
    print(f"  Time fit: {t1 - t0:.2f}s | Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

total_time = time.time() - start_all
if not results:
    print("No models were trained successfully. Exiting.")
    sys.exit(1)

 
out_dir = os.path.join(os.path.dirname(__file__) if '__file__' in globals() else os.getcwd(), 'models')
os.makedirs(out_dir, exist_ok=True)

# Save models and build a summary DataFrame
summary_rows = []
for entry in results:
    nm = entry['name']
    model_file = os.path.join(out_dir, f"model_{nm}.joblib")
    try:
        joblib.dump(entry['model'], model_file)
    except Exception:
        pass
    summary_rows.append({k: v for k, v in entry.items() if k != 'model'})

summary_df = pd.DataFrame(summary_rows)
summary_csv = os.path.join(out_dir, 'summary.csv')
summary_df.to_csv(summary_csv, index=False)

# Rank models by speed (fit_time) and accuracy (test_f1)
summary_df['speed_rank'] = summary_df['fit_time_s'].rank(method='min')
summary_df['accuracy_rank'] = (-summary_df['test_f1']).rank(method='min')

# Short explanations (why) for each model
reasons = {
    'LogisticRegression': 'Fast linear model; good baseline for linearly separable data.',
    'RandomForest': 'Ensemble of trees; robust, handles non-linearities and noisy features.',
    'GaussianNB': 'Very fast probabilistic classifier assuming feature independence.',
    'DecisionTree': 'Single tree; interpretable but prone to overfitting.',
    'ExtraTrees': 'Similar to RandomForest but uses more randomness; often faster to train.'
}

print('\nModel ranking summary:')
print('Rank | Model             | Speed(s) | SpeedRank | Test F1  | AccRank | Reason')
print('-----|-------------------|----------|-----------|----------|---------|-----------------------------------------------------------')
sorted_df = summary_df.sort_values(['accuracy_rank', 'speed_rank'])
for i, row in sorted_df.reset_index(drop=True).iterrows():
    name = row['name']
    print(f"{i+1:4d} | {name:17s} | {row['fit_time_s']:8.2f} | {int(row['speed_rank']):9d} | {row['test_f1']:8.4f} | {int(row['accuracy_rank']):7d} | {reasons.get(name, '')}")

scaler_path = os.path.join(out_dir, "scaler.joblib")
joblib.dump(scaler, scaler_path)
# Save metadata needed for inference: category labels and feature column order
try:
    categories_path = os.path.join(out_dir, 'categories.joblib')
    joblib.dump(list(categories.astype(str)), categories_path)
except Exception:
    categories_path = None

try:
    feature_columns_path = os.path.join(out_dir, 'feature_columns.joblib')
    joblib.dump(list(X.columns), feature_columns_path)
except Exception:
    feature_columns_path = None

# Save the best model separately (best by test_f1)
try:
    best_entry = max(results, key=lambda r: r['test_f1'])
    best_model_file = os.path.join(out_dir, 'best_model.joblib')
    joblib.dump(best_entry['model'], best_model_file)
except Exception:
    best_model_file = None

print(f"\nWrote summary to: {summary_csv}")
print(f"Saved scaler to: {scaler_path}")
if categories_path:
    print(f"Saved category labels to: {categories_path}")
if feature_columns_path:
    print(f"Saved feature column order to: {feature_columns_path}")
if best_model_file:
    print(f"Saved best model to: {best_model_file}")
print(f"Total training time: {total_time:.2f}s")
