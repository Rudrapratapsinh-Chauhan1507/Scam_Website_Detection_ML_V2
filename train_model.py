import os
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import resample
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

warnings.filterwarnings("ignore")

# Config
DATASET_PATH   = "dataset/ml_project_dataset_500.csv"
MODEL_DIR      = "pkl_models"
TFIDF_FEATURES = 50
RANDOM_STATE   = 42
TEST_SIZE      = 0.2

os.makedirs(MODEL_DIR, exist_ok=True)

# Load
df = pd.read_csv(DATASET_PATH)
df["text_content"] = df["text_content"].fillna("").astype(str)

print(f"Dataset shape: {df.shape}")
print(f"Label distribution:\n{df['label'].value_counts()}\n")

# Feature leak note
# 'ssl_valid' and 'has_ssl' are perfectly correlated (identical columns).
# Keeping both inflates feature importance scores. Drop the duplicate.
if "ssl_valid" in df.columns and "has_ssl" in df.columns:
    print("[INFO] Dropping duplicate column 'ssl_valid' (identical to 'has_ssl').\n")
    df = df.drop(columns=["ssl_valid"])

X_raw = df.drop(columns=["label"])
y     = df["label"]

# ── Train / test split ────────────────────────────────────────────────────────
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# ── TF-IDF (fit on train text only — no leakage) ──────────────────────────────
tfidf = TfidfVectorizer(
    max_features=TFIDF_FEATURES,
    stop_words="english",
    max_df=0.85,
    min_df=2,
)

train_text = X_train_raw["text_content"]
test_text  = X_test_raw["text_content"]

tfidf_train = pd.DataFrame(
    tfidf.fit_transform(train_text).toarray(),
    columns=tfidf.get_feature_names_out(),
    index=X_train_raw.index,
)
tfidf_test = pd.DataFrame(
    tfidf.transform(test_text).toarray(),
    columns=tfidf.get_feature_names_out(),
    index=X_test_raw.index,
)

drop_cols = ["url", "text_content"]
X_train_num = X_train_raw.drop(columns=[c for c in drop_cols if c in X_train_raw.columns])
X_test_num  = X_test_raw.drop(columns=[c for c in drop_cols if c in X_test_raw.columns])

X_train_num = X_train_num.reset_index(drop=True)
X_test_num  = X_test_num.reset_index(drop=True)
tfidf_train = tfidf_train.reset_index(drop=True)
tfidf_test  = tfidf_test.reset_index(drop=True)

X_train_full = pd.concat([X_train_num, tfidf_train], axis=1)
X_test_full  = pd.concat([X_test_num,  tfidf_test],  axis=1)

# Remove any remaining duplicate columns
X_train_full = X_train_full.loc[:, ~X_train_full.columns.duplicated()]
X_test_full  = X_test_full.loc[:,  ~X_test_full.columns.duplicated()]

feature_columns = X_train_full.columns.tolist()
y_train = y_train.reset_index(drop=True)

# Impute NaNs (domain_age_days, ssl_days_remaining, domain_expiry_days)
imputer = SimpleImputer(strategy="median")
X_train_imp = pd.DataFrame(
    imputer.fit_transform(X_train_full), columns=feature_columns
)
X_test_imp  = pd.DataFrame(
    imputer.transform(X_test_full), columns=feature_columns
)

# Upsample minority class (train only)
train_df = pd.concat([X_train_imp, y_train], axis=1)
majority  = train_df[train_df["label"] == 0]
minority  = train_df[train_df["label"] == 1]

minority_up = resample(
    minority,
    replace=True,
    n_samples=len(majority),
    random_state=RANDOM_STATE,
)
train_balanced = pd.concat([majority, minority_up]).sample(
    frac=1, random_state=RANDOM_STATE
).reset_index(drop=True)

X_train_bal = train_balanced.drop(columns=["label"])
y_train_bal = train_balanced["label"]

# ── Scale ──────────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = pd.DataFrame(scaler.fit_transform(X_train_bal), columns=feature_columns)
X_test_sc  = pd.DataFrame(scaler.transform(X_test_imp),      columns=feature_columns)

# Also prepare unbalanced-but-imputed+scaled version for honest CV
X_all_imp = pd.DataFrame(
    imputer.transform(
        pd.concat([X_train_num, tfidf_train], axis=1)
        .loc[:, ~pd.concat([X_train_num, tfidf_train], axis=1).columns.duplicated()]
    ),
    columns=feature_columns,
)
X_all_sc  = pd.DataFrame(scaler.transform(X_all_imp), columns=feature_columns)


# Define models
models = {
    "Logistic Regression": LogisticRegression(
        C=0.5,                    # regularisation — fights overfit on small data
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        solver="lbfgs",
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=6,              
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=8,              # cap depth — key fix vs original (max_depth=None)
        min_samples_leaf=4,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
}

# Train, evaluate, cross-validate
cv         = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
results    = {}
best_name  = None
best_score = -1

for name, model in models.items():
    model.fit(X_train_sc, y_train_bal)

    y_pred  = model.predict(X_test_sc)
    y_proba = model.predict_proba(X_test_sc)[:, 1]

    acc     = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    cm      = confusion_matrix(y_test, y_pred)
    report  = classification_report(y_test, y_pred)

    # CV on the original (non-upsampled) training data for an honest estimate
    cv_scores = cross_val_score(
        model, X_train_sc, y_train_bal, cv=cv, scoring="roc_auc"
    )

    results[name] = {
        "model":     model,
        "acc":       acc,
        "roc_auc":   roc_auc,
        "cv_mean":   cv_scores.mean(),
        "cv_std":    cv_scores.std(),
        "cm":        cm,
        "report":    report,
        "y_pred":    y_pred,
        "y_proba":   y_proba,
    }

    print("=" * 55)
    print(f"  {name}")
    print("=" * 55)
    print(f"  Accuracy      : {acc:.4f}")
    print(f"  ROC-AUC       : {roc_auc:.4f}")
    print(f"  5-Fold CV AUC : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print("\n  Confusion Matrix:")
    print(f"  {cm}")
    print("\n  Classification Report:")
    for line in report.strip().split("\n"):
        print(f"  {line}")

    if name == "Random Forest":
        fi = pd.Series(model.feature_importances_, index=feature_columns)
        print("\n  Top 15 Features:")
        for feat, imp in fi.nlargest(15).items():
            print(f"    {feat:<30} {imp:.6f}")

    if name == "Logistic Regression":
        coefs = pd.Series(np.abs(model.coef_[0]), index=feature_columns)
        print("\n  Top 15 Feature Coefficients (|coef|):")
        for feat, val in coefs.nlargest(15).items():
            print(f"    {feat:<30} {val:.4f}")

    print()

    if cv_scores.mean() > best_score:
        best_score = cv_scores.mean()
        best_name  = name

# Summary table
print("=" * 55)
print("  MODEL COMPARISON SUMMARY")
print("=" * 55)
print(f"  {'Model':<25} {'Acc':>6}  {'AUC':>6}  {'CV AUC':>8}")
print(f"  {'-'*25}  {'------'}  {'------'}  {'--------'}")
for name, r in results.items():
    marker = " ← best CV" if name == best_name else ""
    print(f"  {name:<25} {r['acc']:>6.4f}  {r['roc_auc']:>6.4f}  "
          f"{r['cv_mean']:>6.4f}±{r['cv_std']:.4f}{marker}")
print()

# Overfitting diagnostic
print("=" * 55)
print("  OVERFITTING DIAGNOSTIC")
print("=" * 55)
for name, r in results.items():
    gap = r["roc_auc"] - r["cv_mean"]
    flag = "  ⚠ possible overfit" if gap > 0.05 else "  ✓ looks reasonable"
    print(f"  {name:<25}  test AUC - CV AUC = {gap:+.4f}{flag}")
print()

# Save best model artefacts
best_model = results[best_name]["model"]

joblib.dump(tfidf,           f"{MODEL_DIR}/tfidf_vectorizer.pkl")
joblib.dump(best_model,      f"{MODEL_DIR}/scam_detector_model.pkl")
joblib.dump(feature_columns, f"{MODEL_DIR}/feature_columns.pkl")
joblib.dump(scaler,          f"{MODEL_DIR}/scaler.pkl")
joblib.dump(imputer,         f"{MODEL_DIR}/imputer.pkl")

# Save all three models individually too
for name, r in results.items():
    fname = name.lower().replace(" ", "_")
    joblib.dump(r["model"], f"{MODEL_DIR}/model_{fname}.pkl")
    print(f"[OK] Saved  {MODEL_DIR}/model_{fname}.pkl")

print(f"\n[BEST] '{best_name}' selected (CV AUC = {best_score:.4f})")
print(f"[OK]  Best model saved as  {MODEL_DIR}/scam_detector_model.pkl")
print(f"[OK]  Imputer  saved as    {MODEL_DIR}/imputer.pkl")
print(f"[OK]  All artefacts saved to {MODEL_DIR}/")