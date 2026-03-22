import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import resample

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score


df = pd.read_csv("dataset/ml_project_dataset.csv")

df["text_content"] = df['text_content'].fillna("").astype(str)

# balance dataset
df_majority = df[df['label'] == 0]
df_minority = df[df['label'] == 1]

df_minority_upsampled = resample(
    df_minority,
    replace=True,
    n_samples=len(df_majority),
    random_state=42
)

df = pd.concat([df_majority, df_minority_upsampled])

# tf-idf
tfidf = TfidfVectorizer(max_features=50, stop_words="english", max_df=0.85, min_df=2)

tfidf_matrix = tfidf.fit_transform(df['text_content'])

tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())

df = df.drop(columns=["url", 'text_content'])

df = df.reset_index(drop=True)
tfidf_df = tfidf_df.reset_index(drop=True)

df = df.loc[:, ~df.columns.duplicated()]
tfidf_df = tfidf_df.loc[:, ~tfidf_df.columns.duplicated()]

df = pd.concat([df, tfidf_df], axis=1)

# Separate features and label
X = df.drop(columns=["label"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scaling
scaler = StandardScaler()
X_train = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X.columns
)

X_test = pd.DataFrame(
    scaler.transform(X_test),
    columns=X.columns
)

model_rf = RandomForestClassifier(n_estimators=2000, random_state=42)
model_rf.fit(X_train, y_train)

y_pred_rf = model_rf.predict(X_test)

print("Random Forest Accuracy:")
print(accuracy_score(y_test, y_pred_rf))

print("\nRF Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

print("\nRF Classification Report:")
print(classification_report(y_test, y_pred_rf))

print("\nRF ROC AUC Score:")
print(roc_auc_score(y_test, model_rf.predict_proba(X_test)[:,1]))

joblib.dump(tfidf, "./pkl_models/tfidf_vectorizer.pkl")
joblib.dump(model_rf, "./pkl_models/scam_detector_model.pkl")
joblib.dump(X.columns.tolist(), "./pkl_models/feature_columns.pkl")
joblib.dump(scaler, "./pkl_models/scaler.pkl")

print("\nModel and feature columns saved successfully.")

# # Logistic Regression
# model = LogisticRegression(max_iter=2000, class_weight="balanced")
# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)

# Logistic Regression
# print("Logistic Regression Accuracy:")
# print(accuracy_score(y_test, y_pred))

# print("\nConfusion Matrix:")
# print(confusion_matrix(y_test, y_pred))

# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))

# print("\nROC AUC Score:")
# print(roc_auc_score(y_test, model.predict_proba(X_test)[:,1]))