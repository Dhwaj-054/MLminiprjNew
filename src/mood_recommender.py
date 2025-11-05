"""
Mood-based Song Recommender

Single-file organized script that:
- Loads `data_moods.csv` from project root
- Performs EDA and saves key visualizations
- Preprocesses features (missing values, outlier removal, scaling)
- Trains RandomForest, XGBoost, LightGBM with 5-fold CV and randomized hyperparameter search
- Compares models (accuracy, precision/recall/f1, confusion matrix) and saves best model
- Provides a `recommend_songs(mood_str)` function and an interactive loop

Designed for clarity and production-readiness with error handling and logging.
"""

import os
import time
import logging
from pathlib import Path
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data_moods.csv"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)


def load_data(path=DATA_PATH):
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(path)
    logging.info(f"Loaded data with shape: {df.shape}")
    return df


def perform_eda(df: pd.DataFrame):
    """Print basic EDA insights and save some visualizations."""
    logging.info("Performing EDA...")
    print("Columns:", df.columns.tolist())
    print("Sample rows:")
    print(df.head())

    # Missing values
    missing = df.isnull().sum().sort_values(ascending=False)
    print("Missing values (top 10):")
    print(missing.head(10))

    # Numeric summary
    num = df.select_dtypes(include=[np.number])
    print("Numeric summary:")
    print(num.describe().T)

    # Correlation heatmap for numeric features
    if num.shape[1] > 1:
        plt.figure(figsize=(10, 8))
        sns.heatmap(num.corr(), annot=True, fmt='.2f', cmap='coolwarm')
        plt.title('Numeric feature correlation')
        p = OUTPUT_DIR / 'correlation_heatmap.png'
        plt.savefig(p, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved correlation heatmap to {p}")


def preprocess(df: pd.DataFrame, target_col='mood'):
    """Handle missing values, outliers, encode labels, split dataset.

    Returns: X_train, X_test, y_train, y_test, full_df, label_encoder, scaler
    """
    logging.info("Starting preprocessing...")
    df_ = df.copy()

    # Basic cleaning: drop duplicates
    before = len(df_)
    df_.drop_duplicates(inplace=True)
    logging.info(f"Dropped {before - len(df_)} duplicate rows")

    # Select numeric features commonly present in Spotify datasets
    numeric_cols = [
        c for c in df_.columns if df_[c].dtype in ["float64", "int64"] and c != 'id'
    ]

    # If target is not present, raise
    if target_col not in df_.columns:
        raise KeyError(f"Target column '{target_col}' not found in data")

    # Handle missing values: numeric -> median, categorical -> mode
    for c in numeric_cols:
        if df_[c].isnull().any():
            df_[c].fillna(df_[c].median(), inplace=True)

    # Outlier removal using IQR on numeric columns
    for c in numeric_cols:
        q1 = df_[c].quantile(0.25)
        q3 = df_[c].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        before = len(df_)
        df_ = df_[(df_[c] >= lower) & (df_[c] <= upper)]
        after = len(df_)
        if before != after:
            logging.info(f"Removed {before-after} outliers from {c}")

    # Encode label
    le = LabelEncoder()
    df_ = df_.reset_index(drop=True)
    df_['mood_label'] = le.fit_transform(df_[target_col].astype(str))

    # Define features: prefer audio features if present
    feature_candidates = [
        'tempo', 'energy', 'valence', 'danceability', 'speechiness', 'acousticness', 'instrumentalness'
    ]
    features = [c for c in feature_candidates if c in df_.columns]
    if not features:
        # fallback to numeric columns excluding the label
        features = [c for c in numeric_cols if c != df_['mood_label']]

    X = df_[features]
    y = df_['mood_label']

    # Train/test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logging.info(f"Preprocessing done. Features used: {features}")
    return X_train_scaled, X_test_scaled, y_train, y_test, df_, le, scaler, features


def train_and_evaluate(X_train, X_test, y_train, y_test, models_to_run=None):
    """Train three models, run randomized search, compute metrics and pick best model.

    Returns dict of results and best_model (fitted with best hyperparams)
    """
    logging.info("Starting training and model evaluation...")
    results = {}

    if models_to_run is None:
        models_to_run = ['random_forest', 'xgboost', 'lightgbm']

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Define model constructors and search spaces
    model_defs = {}

    model_defs['random_forest'] = {
        'estimator': RandomForestClassifier(n_jobs=-1, random_state=42),
        'param_dist': {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
        },
    }

    if XGBClassifier is not None:
        model_defs['xgboost'] = {
            'estimator': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', verbosity=0),
            'param_dist': {
                'n_estimators': [100, 200],
                'max_depth': [3, 6],
                'learning_rate': [0.05, 0.1],
            },
        }

    if LGBMClassifier is not None:
        model_defs['lightgbm'] = {
            'estimator': LGBMClassifier(random_state=42),
            'param_dist': {
                'n_estimators': [100, 200],
                'num_leaves': [31, 50],
                'learning_rate': [0.05, 0.1],
            },
        }

    best_overall = {'name': None, 'accuracy': 0.0, 'f1': 0.0, 'model': None}

    for name in models_to_run:
        if name not in model_defs:
            logging.warning(f"Skipping {name} because constructor not available")
            continue

        estimator = model_defs[name]['estimator']
        param_dist = model_defs[name]['param_dist']

        rs = RandomizedSearchCV(
            estimator,
            param_distributions=param_dist,
            n_iter=6,
            cv=skf,
            scoring='accuracy',
            n_jobs=-1,
            random_state=42,
            verbose=0,
        )

        t0 = time.time()
        rs.fit(X_train, y_train)
        train_time = time.time() - t0

        best = rs.best_estimator_
        y_pred = best.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
            'best_params': rs.best_params_,
            'accuracy': acc,
            'precision': p,
            'recall': r,
            'f1': f1,
            'confusion_matrix': cm,
            'train_time': train_time,
            'model': best,
        }

        logging.info(f"{name} done: acc={acc:.4f}, f1={f1:.4f}, time={train_time:.1f}s")

        # update best
        if (acc > best_overall['accuracy']) or (acc == best_overall['accuracy'] and f1 > best_overall['f1']):
            best_overall.update({'name': name, 'accuracy': acc, 'f1': f1, 'model': best})

    return results, best_overall


def plot_model_comparison(results: dict):
    names = []
    accs = []
    f1s = []
    times = []
    for k, v in results.items():
        names.append(k)
        accs.append(v['accuracy'])
        f1s.append(v['f1'])
        times.append(v['train_time'])

    # Accuracy & F1 bar chart
    x = np.arange(len(names))
    width = 0.35
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, accs, width, label='Accuracy')
    plt.bar(x + width/2, f1s, width, label='F1 (macro)')
    plt.xticks(x, names)
    plt.legend()
    plt.ylabel('Score')
    plt.title('Model comparison')
    p = OUTPUT_DIR / 'model_comparison.png'
    plt.savefig(p, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved model comparison plot to {p}")

    # Training time
    plt.figure(figsize=(8, 4))
    plt.bar(names, times)
    plt.ylabel('Training time (s)')
    plt.title('Training time comparison')
    p2 = OUTPUT_DIR / 'training_time.png'
    plt.savefig(p2, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved training time plot to {p2}")


def show_confusion_matrix(cm, classes, outpath):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig(outpath, bbox_inches='tight')
    plt.close()


def recommend_songs(df_full: pd.DataFrame, mood_str: str, label_encoder: LabelEncoder, features, top_k=5):
    """Return top_k songs matching the mood_str. If mood_str can't be encoded, tries a fuzzy match.

    Recommendation strategy: filter songs whose mood equals requested mood and sort by a composite score using energy and valence and tempo (if present).
    """
    # encode mood
    moods = list(label_encoder.classes_)
    mood_str = str(mood_str).strip().lower()
    # find best matching mood class
    matched = None
    for m in moods:
        if m.lower() == mood_str:
            matched = m
            break

    if matched is None:
        # try substring match
        for m in moods:
            if mood_str in m.lower() or m.lower() in mood_str:
                matched = m
                break

    if matched is None:
        raise ValueError(f"Could not map input '{mood_str}' to any known mood: {moods}")

    label = int(label_encoder.transform([matched])[0])
    candidates = df_full[df_full['mood_label'] == label].copy()
    if candidates.empty:
        return []

    # compute a simple composite score
    score = np.zeros(len(candidates))
    if 'energy' in features:
        score += candidates['energy'].fillna(0).values
    if 'valence' in features:
        score += candidates['valence'].fillna(0).values
    if 'tempo' in features:
        # normalize tempo scale before adding
        score += (candidates['tempo'].fillna(candidates['tempo'].median()).values / 200.0)

    candidates['_score'] = score
    recs = candidates.sort_values('_score', ascending=False).head(top_k)

    # Select useful columns if present
    cols = [c for c in ['track_name', 'artist', 'mood'] + features if c in recs.columns]
    return recs[cols]


def save_model(model, scaler, label_encoder, features, path=MODEL_DIR / 'best_model.joblib'):
    payload = {
        'model': model,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'features': features,
    }
    joblib.dump(payload, path)
    logging.info(f"Saved best model bundle to {path}")


def save_results(results, path=MODEL_DIR / 'results.joblib'):
    """Save training results (metrics + models) for use by the Streamlit app."""
    try:
        joblib.dump(results, path)
        logging.info(f"Saved training results to {path}")
    except Exception as e:
        logging.error(f"Could not save results: {e}")


def train_and_save_all(target_col='mood'):
    """Convenience wrapper to train models from raw CSV, save results and best model bundle.

    Returns (results, best_overall, df_full, label_encoder, features)
    """
    df = load_data()
    X_train, X_test, y_train, y_test, df_full, le, scaler, features = preprocess(df, target_col=target_col)
    results, best = train_and_evaluate(X_train, X_test, y_train, y_test)

    # Save results and best model
    save_results(results)
    if best.get('model') is not None:
        save_model(best['model'], scaler, le, features)

    return results, best, df_full, le, features


def interactive_loop(df_full, label_encoder, features, best_model_bundle):
    model = best_model_bundle
    print("Enter a mood (e.g., 'happy', 'sad', 'energetic') or 'quit' to exit.")
    while True:
        s = input('Mood> ').strip()
        if s.lower() in ('quit', 'exit'):
            break
        try:
            recs = recommend_songs(df_full, s, label_encoder, features, top_k=5)
            if recs is None or recs.empty:
                print(f"No recommendations found for '{s}'")
                continue
            print(f"Top recommendations for mood '{s}':")
            print(recs.to_string(index=False))
        except Exception as e:
            print(str(e))


def main():
    df = load_data()
    perform_eda(df)

    X_train, X_test, y_train, y_test, df_full, le, scaler, features = preprocess(df, target_col='mood')

    results, best = train_and_evaluate(X_train, X_test, y_train, y_test)

    # Print model comparison table
    comparison = pd.DataFrame([{
        'model': k,
        'accuracy': v['accuracy'],
        'precision': v['precision'],
        'recall': v['recall'],
        'f1': v['f1'],
        'train_time_s': v['train_time'],
    } for k, v in results.items()])
    print("Model comparison:\n", comparison.sort_values('accuracy', ascending=False))

    # Save plots
    plot_model_comparison(results)

    # Save confusion matrices
    for k, v in results.items():
        try:
            classes = list(le.classes_)
            p = OUTPUT_DIR / f'confusion_{k}.png'
            show_confusion_matrix(v['confusion_matrix'], classes, p)
        except Exception:
            pass

    print("Best model:", best['name'], f"(acc={best['accuracy']:.4f}, f1={best['f1']:.4f})")

    # Feature importance if available
    try:
        model = best['model']
        if hasattr(model, 'feature_importances_'):
            fi = model.feature_importances_
            for f, val in zip(features, fi):
                print(f"Feature importance - {f}: {val:.4f}")
    except Exception:
        pass

    # Save best model bundle
    if best['model'] is not None:
        save_model(best['model'], scaler, le, features)

    # Start interactive loop
    interactive_loop(df_full, le, features, best['model'])


if __name__ == '__main__':
    main()
