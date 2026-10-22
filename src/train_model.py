"""Train ETA-delay regression and delay-risk classification models."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from features import FEATURES, REGRESSION_TARGET, CLASSIFICATION_TARGET, validate_features

ROOT = Path(__file__).resolve().parents[1]


def build_preprocessor(df, features):
    # Detect categorical columns by their actual data type.
    categorical = df[features].select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    numeric = [c for c in features if c not in categorical]

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]),
                numeric,
            ),
            (
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical,
            ),
        ],
        remainder="drop",
    )


def run_training(data_path=None):
    data_path = Path(data_path) if data_path else ROOT / "data" / "shipments.csv"
    df = pd.read_csv(data_path)
    validate_features(FEATURES)
    X = df[FEATURES].copy()
    y = df[REGRESSION_TARGET].astype(float)
    yc = df[CLASSIFICATION_TARGET].astype(int)
    # One fixed split shared across both tasks, for comparable evaluation.
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(indices, test_size=.2, random_state=42, stratify=yc if yc.nunique() > 1 else None)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    yc_train, yc_test = yc.iloc[train_idx], yc.iloc[test_idx]

    # Strong, interpretable baseline: median delay learned only from training rows.
    baseline_value = float(y_train.median())
    baseline_pred = np.full(len(y_test), baseline_value)

    reg = Pipeline([("preprocess", build_preprocessor(X_train, FEATURES)),
                    ("model", RandomForestRegressor(n_estimators=350, min_samples_leaf=2, max_features=.9, random_state=42, n_jobs=-1))])
    reg.fit(X_train, y_train)
    pred = reg.predict(X_test)
    reg_metrics = {"baseline_median_mae_hours": round(float(mean_absolute_error(y_test, baseline_pred)), 3),
                   "mae_hours": round(float(mean_absolute_error(y_test, pred)), 3),
                   "rmse_hours": round(float(np.sqrt(mean_squared_error(y_test, pred))), 3),
                   "r2": round(float(r2_score(y_test, pred)), 3)}

    clf = Pipeline([("preprocess", build_preprocessor(X_train, FEATURES)),
                    ("model", RandomForestClassifier(n_estimators=350, min_samples_leaf=2, class_weight="balanced_subsample", random_state=42, n_jobs=-1))])
    clf.fit(X_train, yc_train)
    cpred = clf.predict(X_test)
    proba = clf.predict_proba(X_test)[:, list(clf.named_steps["model"].classes_).index(1)] if 1 in clf.named_steps["model"].classes_ else np.zeros(len(X_test))
    cls_metrics = {"precision": round(float(precision_score(yc_test, cpred, zero_division=0)), 3),
                   "recall": round(float(recall_score(yc_test, cpred, zero_division=0)), 3),
                   "f1": round(float(f1_score(yc_test, cpred, zero_division=0)), 3),
                   "roc_auc": round(float(roc_auc_score(yc_test, proba)), 3) if yc_test.nunique() > 1 else None,
                   "confusion_matrix_tn_fp_fn_tp": confusion_matrix(yc_test, cpred, labels=[0,1]).ravel().tolist()}

    model_dir = ROOT / "models"; model_dir.mkdir(exist_ok=True)
    joblib.dump(reg, model_dir / "eta_delay_regressor.joblib")
    joblib.dump(clf, model_dir / "delay_risk_classifier.joblib")
    metrics = {"dataset_rows": int(len(df)), "train_rows": int(len(train_idx)), "test_rows": int(len(test_idx)),
               "delay_threshold_hours": 24, "regression": reg_metrics, "classification": cls_metrics,
               "feature_columns": FEATURES, "prediction_checkpoint": "after destination port arrival, before customs clearance"}
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Permutation importance is computed on held-out data; robust to preprocessing pipeline.
    imp = permutation_importance(reg, X_test, y_test, n_repeats=8, random_state=42, scoring="neg_mean_absolute_error", n_jobs=-1)
    importance = pd.DataFrame({"feature": FEATURES, "importance_mae_increase_hours": imp.importances_mean, "std": imp.importances_std}).sort_values("importance_mae_increase_hours", ascending=False)
    importance.to_csv(model_dir / "permutation_importance.csv", index=False)
    pred_df = df.iloc[test_idx][["shipment_id", "route", "final_delay_hours", "is_delayed_24h"]].copy()
    pred_df["predicted_delay_hours"] = pred.round(2)
    pred_df["predicted_delay_probability"] = proba.round(3)
    pred_df.to_csv(model_dir / "test_predictions.csv", index=False)
    print(json.dumps(metrics, indent=2))
    print("\nTop held-out permutation importance:\n", importance.head(10).to_string(index=False))
    return metrics, importance

if __name__ == "__main__":
    run_training()
