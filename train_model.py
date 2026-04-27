import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


DEFAULT_DATASET = Path("dataset") / "reference_training_dataset_5000.csv"
DEFAULT_CLEAN_DATASET = Path("dataset") / "reference_training_dataset_5000_clean.csv"
MODEL_DIR = Path("pkl_models")
RANDOM_STATE = 42

METADATA_COLUMNS = {
    "url",
    "final_url",
    "source_dataset",
    "has_reference_text",
    "scrape_status",
    "scrape_error",
    "redirect_count",
    "registrar",
}


def load_clean_dataset(path: Path, clean_output: Path | None = None) -> tuple[pd.DataFrame, list[str]]:
    frame = pd.read_csv(path)
    if "label" not in frame.columns:
        raise ValueError("Dataset must contain a label column.")

    frame["label"] = pd.to_numeric(frame["label"], errors="coerce")
    frame = frame[frame["label"].isin([0, 1])].copy()
    frame["label"] = frame["label"].astype(int)

    drop_columns = [column for column in METADATA_COLUMNS if column in frame.columns]
    feature_frame = frame.drop(columns=drop_columns)

    numeric_columns = []
    for column in feature_frame.columns:
        if column == "label":
            continue
        converted = pd.to_numeric(feature_frame[column], errors="coerce")
        if converted.notna().any() and converted.nunique(dropna=False) > 1:
            feature_frame[column] = converted
            numeric_columns.append(column)

    clean = feature_frame[["label", *numeric_columns]].copy()
    if clean_output is not None:
        clean_output.parent.mkdir(parents=True, exist_ok=True)
        clean.to_csv(clean_output, index=False, encoding="utf-8")

    return clean, numeric_columns


def candidate_models() -> dict[str, Pipeline]:
    tree_prep = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    scaled_prep = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    return {
        "extra_trees": Pipeline(
            [
                ("prep", tree_prep),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=700,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("prep", tree_prep),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("prep", tree_prep),
                ("model", GradientBoostingClassifier(random_state=RANDOM_STATE)),
            ]
        ),
        "logistic_regression": Pipeline(
            [
                ("prep", scaled_prep),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "svc_rbf": Pipeline(
            [
                ("prep", scaled_prep),
                (
                    "model",
                    SVC(
                        C=2.0,
                        gamma="scale",
                        probability=True,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def evaluate_candidates(X_train: pd.DataFrame, y_train: pd.Series) -> list[dict]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    results = []
    for name, model in candidate_models().items():
        scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
        result = {"model": name}
        for metric in scoring:
            values = scores[f"test_{metric}"]
            result[f"{metric}_mean"] = float(values.mean())
            result[f"{metric}_std"] = float(values.std())
        results.append(result)
        print(
            f"[CV] {name}: f1={result['f1_mean']:.4f}, "
            f"auc={result['roc_auc_mean']:.4f}, acc={result['accuracy_mean']:.4f}"
        )

    return sorted(results, key=lambda item: (item["f1_mean"], item["roc_auc_mean"]), reverse=True)


def holdout_metrics(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    predictions = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
    }
    if hasattr(model, "predict_proba"):
        metrics["roc_auc"] = float(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
    return metrics


def train(args: argparse.Namespace) -> None:
    dataset_path = Path(args.dataset)
    clean_path = Path(args.clean_output)
    clean, feature_columns = load_clean_dataset(dataset_path, clean_path)

    X = clean[feature_columns]
    y = clean["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    cv_results = evaluate_candidates(X_train, y_train)
    best_name = cv_results[0]["model"]
    best_model = candidate_models()[best_name]
    best_model.fit(X_train, y_train)

    final_model = best_model
    if args.calibrate:
        final_model = CalibratedClassifierCV(best_model, method="isotonic", cv=3)
        final_model.fit(X_train, y_train)

    metrics = holdout_metrics(final_model, X_test, y_test)
    print(f"[BEST] {best_name}")
    print(f"[HOLDOUT] {metrics}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "best_scam_detector.pkl"
    features_path = MODEL_DIR / "model_features.pkl"
    report_path = MODEL_DIR / "training_report.json"

    joblib.dump(final_model, model_path)
    joblib.dump(feature_columns, features_path)

    report = {
        "dataset": str(dataset_path),
        "clean_dataset": str(clean_path),
        "rows": int(len(clean)),
        "features": feature_columns,
        "best_model": best_name,
        "calibrated": bool(args.calibrate),
        "cv_results": cv_results,
        "holdout_metrics": metrics,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[OK] Saved model: {model_path}")
    print(f"[OK] Saved features: {features_path}")
    print(f"[OK] Saved report: {report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the scam detector ML model.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--clean-output", default=str(DEFAULT_CLEAN_DATASET))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--no-calibrate",
        action="store_false",
        dest="calibrate",
        help="Save the best model without probability calibration.",
    )
    parser.set_defaults(calibrate=True)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
