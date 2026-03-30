import os
import re
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from src.tfidf_vectorizer import TFIDFFeatureExtractor

# ----------------------------------------
# CONFIG
# ----------------------------------------
DATA_PATH = "./dataset/Scam_website_dataset.csv"
MODEL_DIR = "./pkl_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ----------------------------------------
# LOAD DATA
# ----------------------------------------
df = pd.read_csv(DATA_PATH)
print(f"[INFO] Dataset loaded: {df.shape}")

# ----------------------------------------
# TEXT CLEANING
# ----------------------------------------
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else "empty_doc"

df["text_content"] = df["text_content"].fillna("").apply(clean_text)

# Remove weak samples
df = df[df["text_content"].str.len() > 10]

# Fix labels
df["label"] = pd.to_numeric(df["label"], errors="coerce")
df = df[df["label"].isin([0, 1])]
df["label"] = df["label"].astype(int)

print(f"[INFO] Final dataset: {df.shape}")
print(f"[INFO] Label distribution:\n{df['label'].value_counts()}")

# ----------------------------------------
# FEATURE PREP
# ----------------------------------------
drop_cols = [
    "url", "text_content", "label",
    "id", "title", "meta_description",
    "final_url", "status", "error_message",
    "scraped_at", "registrar",
]

X_struct = df.drop(columns=drop_cols, errors="ignore")
texts = df["text_content"]
y = df["label"]

X_struct = X_struct.apply(pd.to_numeric, errors="coerce").fillna(0)

# ----------------------------------------
# REMOVE LEAKAGE FEATURES
# ----------------------------------------
leak_cols = [
    "domain_age_days", "domain_expiry_days",
    "ssl_days_remaining", "short_expiry_domain", "is_new_domain"
]
X_struct.drop(columns=[c for c in leak_cols if c in X_struct.columns], inplace=True, errors="ignore")

# Remove overfitting features
X_struct.drop(columns=[
    c for c in ["avg_word_length", "scam_keyword_count", "scam_keyword_density"]
    if c in X_struct.columns
], inplace=True, errors="ignore")

struct_columns = list(X_struct.columns)

# ----------------------------------------
# CROSS VALIDATION
# ----------------------------------------
cv_scores = cross_val_score(
    RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    X_struct, y, cv=5
)

print(f"\n[CV] Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ----------------------------------------
# TRAIN TEST SPLIT
# ----------------------------------------
X_train_s, X_test_s, text_train, text_test, y_train, y_test = train_test_split(
    X_struct, texts, y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# ----------------------------------------
# TF-IDF
# ----------------------------------------
tfidf = TFIDFFeatureExtractor(max_features=1200)

X_train_text = tfidf.fit_transform(text_train.tolist()).toarray()
X_test_text  = tfidf.transform(text_test.tolist()).toarray()

tfidf_columns = [f"tfidf_{f}" for f in tfidf.vectorizer.get_feature_names_out()]

# Combine
X_train = np.hstack([X_train_s.values, X_train_text])
X_test  = np.hstack([X_test_s.values, X_test_text])

all_columns = struct_columns + tfidf_columns

print(f"[INFO] Features: {len(all_columns)} (Struct: {len(struct_columns)}, TF-IDF: {len(tfidf_columns)})")

# ----------------------------------------
# SCALING
# ----------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ----------------------------------------
# LOGISTIC REGRESSION
# ----------------------------------------
lr = LogisticRegression(max_iter=5000, class_weight="balanced", C=0.5, random_state=42)
lr.fit(X_train_scaled, y_train)

y_pred_lr = lr.predict(X_test_scaled)
acc_lr = accuracy_score(y_test, y_pred_lr)

# ----------------------------------------
# RANDOM FOREST
# ----------------------------------------
rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)

# ----------------------------------------
# RESULTS
# ----------------------------------------
print("\n===== MODEL RESULTS =====")

print("\n[Logistic Regression]")
print(f"Accuracy : {acc_lr:.4f}")

print("\n[Random Forest]")
print(f"Accuracy : {acc_rf:.4f}")

# ----------------------------------------
# BEST MODEL
# ----------------------------------------
if acc_rf >= acc_lr:
    best_model = rf
    model_name = "random_forest"
    use_scaler = False
else:
    best_model = lr
    model_name = "logistic_regression"
    use_scaler = True

print(f"\n[INFO] Best model: {model_name}")

# ----------------------------------------
# SAVE
# ----------------------------------------
joblib.dump(best_model, f"{MODEL_DIR}/{model_name}.pkl")
joblib.dump(all_columns, f"{MODEL_DIR}/feature_columns.pkl")
tfidf.save(f"{MODEL_DIR}/tfidf_vectorizer.pkl")

if use_scaler:
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")

print("\n[OK] Model pipeline saved successfully")