import os
import re
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

from src.tfidf_vectorizer import TFIDFFeatureExtractor

# ================= CONFIG =================
DATA_PATH = "./dataset/Scam_website_dataset.csv"
MODEL_DIR = "./pkl_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ================= LOAD =================
df = pd.read_csv(DATA_PATH)
print(f"[INFO] Dataset: {df.shape}")

# ================= CLEAN TEXT =================
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else "empty_doc"

df["text_content"] = df["text_content"].fillna("").apply(clean_text)

# ================= LABEL =================
df["label"] = pd.to_numeric(df["label"], errors="coerce")
df = df[df["label"].isin([0, 1])]
df["label"] = df["label"].astype(int)

print(df["label"].value_counts())

# ================= FEATURES =================
y = df["label"]
texts = df["text_content"]

drop_cols = ["label", "text_content", "url", "id", "scraped_at", "error_message"]
X_struct = df.drop(columns=drop_cols, errors="ignore")

X_struct = X_struct.apply(pd.to_numeric, errors="coerce").fillna(0)
struct_columns = list(X_struct.columns)

# ================= SPLIT =================
X_train_s, X_test_s, text_train, text_test, y_train, y_test = train_test_split(
    X_struct, texts, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ================= TF-IDF (REDUCED) =================
tfidf = TFIDFFeatureExtractor(max_features=300)

X_train_text = tfidf.fit_transform(text_train.tolist()).toarray()
X_test_text = tfidf.transform(text_test.tolist()).toarray()

tfidf_columns = [f"tfidf_{f}" for f in tfidf.vectorizer.get_feature_names_out()]

# ================= COMBINE =================
X_train = np.hstack([X_train_s.values, X_train_text])
X_test = np.hstack([X_test_s.values, X_test_text])

all_columns = struct_columns + tfidf_columns

print(f"[INFO] Features: {len(all_columns)}")

# ================= SCALING =================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ================= MODELS =================
lr = LogisticRegression(max_iter=5000, class_weight="balanced", C=1.0)

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

# ================= CALIBRATION =================
rf_cal = CalibratedClassifierCV(rf, method="sigmoid", cv=3)
lr_cal = CalibratedClassifierCV(lr, method="sigmoid", cv=3)

# ================= TRAIN =================
rf_cal.fit(X_train, y_train)
lr_cal.fit(X_train_scaled, y_train)

# ================= EVAL =================
y_pred_rf = rf_cal.predict(X_test)
y_pred_lr = lr_cal.predict(X_test_scaled)

acc_rf = accuracy_score(y_test, y_pred_rf)
acc_lr = accuracy_score(y_test, y_pred_lr)

print("\n===== RESULTS =====")
print(f"RF Accuracy: {acc_rf:.4f}")
print(f"LR Accuracy: {acc_lr:.4f}")

print("\n[RF Report]\n", classification_report(y_test, y_pred_rf))
print("\n[LR Report]\n", classification_report(y_test, y_pred_lr))

# ================= BEST =================
if acc_rf >= acc_lr:
    best_model = rf_cal
    model_name = "random_forest"
    use_scaler = False
else:
    best_model = lr_cal
    model_name = "logistic_regression"
    use_scaler = True

print(f"\n[INFO] Best model: {model_name}")

# ================= SAVE =================
joblib.dump(best_model, f"{MODEL_DIR}/{model_name}.pkl")
joblib.dump(all_columns, f"{MODEL_DIR}/feature_columns.pkl")
tfidf.save(f"{MODEL_DIR}/tfidf_vectorizer.pkl")

if use_scaler:
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")

print("\n[OK] V2 Model saved")