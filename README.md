# ScamShield - Scam Website Detection ML

ScamShield is an end-to-end web application for detecting suspicious, phishing, and scam websites. It combines live website crawling, URL/content feature extraction, calibrated machine learning scoring, and evidence-backed triage in a professional Flask web console.

The current saved detector is a calibrated **Extra Trees** classifier trained on a balanced 5000-row reference dataset.

## Key Features

- Live URL scanning from the web interface
- URL normalization and redirect-aware crawling
- ML-based scam probability scoring
- Risk band, confidence, and crawl status reporting
- Evidence stack with top model drivers
- Feature-family signal visualization
- Raw feature audit table
- Bulk URL review workflow
- Session case history
- URL mutation lab for testing suspicious patterns
- CLI prediction, training, dataset building, and real-site evaluation scripts

## Current Model Status

```text
Dataset building:       ready
Feature extraction:     ready
Model training:         ready
Saved model artifacts:  ready
Flask web console:      ready
URL prediction CLI:     ready
Real-site evaluation:   ready
```

Latest saved model summary:

```text
Training rows:          5000
Clean training columns: 36
Model features:         35
Best model:             extra_trees
Probability calibrated: yes
Holdout accuracy:       0.9670
Holdout ROC AUC:        0.9941
Real-site test:         43/43 correct
```

## Web Application

The main webapp is the **ScamShield Enterprise Risk Console**.

It provides these modules:

- **Live Scan**: submit a suspicious URL and receive a scam/legitimate verdict.
- **Model Health**: view accuracy, precision, recall, ROC AUC, and feature families.
- **Bulk Review**: paste multiple URLs and scan them in sequence.
- **Case History**: review URLs scanned during the current Flask server session.
- **URL Lab**: generate URL-only mutations and compare risk changes.

Run the webapp:

```powershell
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

Important: `app.py` currently runs with `debug=False`. If you edit `templates/index.html`, restart the Flask server to see your template changes. If CSS or JavaScript changes do not appear, hard refresh the browser with `Ctrl + F5`.

## Web API

The Flask app exposes these endpoints:

```text
GET  /                 Web console
GET  /api/health       Model readiness and summary
GET  /api/model        Model metadata and feature list
GET  /api/history      Session scan history
POST /api/predict      Live crawl + full ML prediction
POST /api/features     URL-only feature preview
POST /api/url-predict  URL-only prediction for fast simulations
```

Example prediction request:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:5000/api/predict -ContentType "application/json" -Body '{"url":"https://example.com","timeout":6}'
```

Prediction output includes:

```text
Final URL after redirects
Scrape status
Prediction: LEGIT or SCAM
Confidence
Scam probability
Risk band
Observed signals
Top model drivers
Raw model feature values
```

## Project Structure

```text
.
+-- app.py
+-- templates/
|   +-- index.html
+-- static/
|   +-- css/
|   |   +-- styles.css
|   +-- js/
|       +-- app.js
+-- dataset/
|   +-- reference_training_dataset_5000.csv
|   +-- reference_training_dataset_5000_clean.csv
|   +-- real_site_prediction_results.csv
+-- pkl_models/
|   +-- best_scam_detector.pkl
|   +-- model_features.pkl
|   +-- training_report.json
+-- src/
|   +-- scraper.py
|   +-- url_features.py
|   +-- content_features.py
|   +-- domain_features.py
|   +-- database_mysql.py
+-- build_reference_dataset.py
+-- train_model.py
+-- predict.py
+-- evaluate_real_sites.py
+-- batch_test.py
+-- export_dataset.py
+-- main.py
+-- training_model.ipynb
+-- visualization.ipynb
+-- requirements.txt
+-- README.md
```

## Setup

Use Python 3.12+.

Install dependencies:

```powershell
pip install -r requirements.txt
```

Required saved artifacts:

```text
pkl_models/best_scam_detector.pkl
pkl_models/model_features.pkl
pkl_models/training_report.json
```

If these files are missing, train the model first:

```powershell
python train_model.py
```

## Dataset

Current training dataset files:

```text
dataset/reference_training_dataset_5000.csv
dataset/reference_training_dataset_5000_clean.csv
```

Dataset summary:

```text
Raw dataset:              5000 rows, 54 columns
Clean dataset:            5000 rows, 36 columns
Labels:                   2500 legitimate, 2500 scam
Rows with reference text: 976
URL/domain-only rows:     4024
```

Build the current reference dataset:

```powershell
python build_reference_dataset.py --rows 5000 --output dataset\reference_training_dataset_5000.csv
```

## Model Features

The saved model currently uses 35 numerical features:

```text
url_length
num_dots
num_hyphen
num_slashes
https
subdomains
has_at_symbol
has_double_slash
has_ip
num_underscores
num_percent
num_digits
num_query_params
has_query
path_depth
suspicious_tld
brand_in_url
is_shortened
suspicious_word_count
url_entropy
hostname_length
digit_letter_ratio
num_special_chars
tld_in_path
longest_digit_run
uses_free_hosting
brand_on_free_hosting
brand_domain_mismatch
text_length
token_count
scam_keyword_count
scam_keyword_density
exclamation_count
caps_ratio
avg_word_length
```

The exact feature list is saved in:

```text
pkl_models/model_features.pkl
```

## Training

Run training:

```powershell
python train_model.py
```

Equivalent explicit command:

```powershell
python train_model.py --dataset dataset\reference_training_dataset_5000.csv --clean-output dataset\reference_training_dataset_5000_clean.csv
```

Candidate models compared during training:

- Extra Trees
- Random Forest
- Gradient Boosting
- SVC RBF
- Logistic Regression

The final model is selected using cross-validated F1 score. The selected model is calibrated with isotonic probability calibration by default.

Saved training report:

```text
pkl_models/training_report.json
```

Holdout performance:

```text
Accuracy:  0.9670
Precision: 0.9736
Recall:    0.9600
F1:        0.9668
ROC AUC:   0.9941
```

Saved artifacts:

```text
pkl_models/best_scam_detector.pkl
pkl_models/model_features.pkl
pkl_models/training_report.json
```

## CLI Prediction

Predict one URL:

```powershell
python predict.py --url https://example.com
```

Predict with a custom scraping timeout:

```powershell
python predict.py --url https://example.com --timeout 5
```

Predict URLs from a text file:

```powershell
python predict.py --input urls.txt
```

Predict from a CSV file with a `url` column:

```powershell
python predict.py --input urls.csv --output dataset\predictions.csv
```

## Real-Site Evaluation

Run evaluation against known legitimate sites and scam/phishing feeds:

```powershell
python evaluate_real_sites.py --legit-count 23 --scam-count 20 --timeout 5
```

Latest saved result:

```text
Total: 43/43 correct
Legit: 23/23 correct
Scam:  20/20 correct
```

Reports:

```text
real_site_prediction_results.md
dataset/real_site_prediction_results.csv
```

## Main Scripts

`app.py`: Flask web console and API server.

`build_reference_dataset.py`: builds a balanced training CSV from reference datasets.

`train_model.py`: cleans data, compares models, saves the calibrated detector, feature list, and report.

`predict.py`: predicts one URL, a text file of URLs, or a CSV of URLs.

`evaluate_real_sites.py`: evaluates the saved model against legitimate websites and scam/phishing feeds.

`batch_test.py`: runs a small manually defined batch of URLs and writes a markdown report.

`export_dataset.py`: exports labeled website records from MySQL into a CSV dataset.

`main.py`: collects website data, extracts URL/domain/content features, and stores records in MySQL.

## Important Notes

ScamShield outputs should be treated as **risk scores**, not absolute truth. Legitimate websites can sometimes look suspicious, and phishing websites change quickly.

Recommended report wording:

```text
ScamShield performs well on the prepared dataset and the latest saved real-site
evaluation. However, scam and phishing patterns change quickly, so predictions
should be treated as risk scores with supporting evidence rather than final
security decisions.
```
