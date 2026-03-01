import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

# Path to the dataset root
DATASET_ROOT = r"unzipped_Training_and_Testing_Dataset/Training and Testing Dataset"

# Collect all CSV files from all participant folders
def collect_all_csvs(dataset_root):
    all_csvs = glob.glob(os.path.join(dataset_root, 'P_*', '*.csv'))
    return all_csvs

# Load and concatenate all data
def load_all_data(csv_files):
    dfs = []
    for file in csv_files:
        df = pd.read_csv(file)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

if __name__ == "__main__":
    print("Loading CSV files...")
    all_csvs = collect_all_csvs(DATASET_ROOT)
    data = load_all_data(all_csvs)
    print(f"Total samples: {len(data)}")
    print(f"Classes: {sorted(data['activity_label'].unique())}")

    # Features: drop participant_id and activity_label
    X = data.drop(['participant_id', 'activity_label'], axis=1)
    y = data['activity_label']

    # Random split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Train size: {len(X_train)}")
    print(f"Test size: {len(X_test)}\n")

    print("Running Grid Search on SVM...")
    pipe = Pipeline([
        ("clf", SVC())
    ])
    param_grid = {
        'clf__C': [0.01, 0.1, 1, 10, 50, 100],
        'clf__gamma': [0.001, 0.01, 0.1, 1],
        'clf__kernel': ['rbf']
    }
    grid_search = GridSearchCV(pipe, param_grid, cv=3)
    grid_search.fit(X_train, y_train)
    print(f"\nBest parameters found:\n{grid_search.best_params_}\n")

    y_pred = grid_search.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Final Test Accuracy: {acc:.4f}\n")
    print("Classification Report:\n")
    print(classification_report(y_test, y_pred, digits=4))
