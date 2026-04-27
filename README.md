# Scam Website Detection ML

This project detects whether a website URL is likely to be **legitimate** or **scam/phishing** using machine learning features extracted from the URL and, when available, page content.

The current trained model is a calibrated **Extra Trees** classifier trained on a balanced 5000-row reference dataset.

## Current Status

```text
Dataset building:       ready
Feature extraction:     ready
Model training:         ready
Saved model artifacts:  ready
Real-site evaluation:   ready
URL prediction CLI:     ready
```

Latest saved results:

```text
Training rows:          5000
Clean training columns: 36
Model features:         35
Best model:             extra_trees
Probability calibrated: yes
Real-site test:         43/43 correct
```

## Project Structure

```text
.
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
+-- batch_test.py
+-- build_reference_dataset.py
+-- evaluate_real_sites.py
+-- export_dataset.py
+-- main.py
+-- predict.py
+-- train_model.py
+-- training_model.ipynb
+-- real_site_prediction_results.md
+-- requirements.txt
+-- README.md
```

## Setup

Install the Python dependencies:

```powershell
pip install -r requirements.txt
```

The project expects Python 3.12+ based on the dependency file.

## Dataset

The current training dataset files are:

```text
dataset/reference_training_dataset_5000.csv
dataset/reference_training_dataset_5000_clean.csv
```

Dataset summary:

```text
Raw dataset:            5000 rows, 54 columns
Clean dataset:          5000 rows, 36 columns
Labels:                 2500 legitimate, 2500 scam
Rows with reference text: 976
URL/domain-only rows:   4024
Scrape status:          not_scraped for all 5000 rows
```

The raw dataset was generated from reference CSV files under:

```text
C:\Users\lenovo\Desktop\Scam Prac
```

Command for the current dataset size:

```powershell
python build_reference_dataset.py --rows 5000 --output dataset\reference_training_dataset_5000.csv
```

Then `train_model.py` creates the clean dataset by removing metadata and keeping usable numerical feature columns.

## Features Used

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

## Model Training

Training can be run from the script:

```powershell
python train_model.py
```

Equivalent explicit command:

```powershell
python train_model.py --dataset dataset\reference_training_dataset_5000.csv --clean-output dataset\reference_training_dataset_5000_clean.csv
```

The project also includes `training_model.ipynb`, which records the training workflow in notebook form.

Models compared during training:

- Extra Trees
- Random Forest
- Gradient Boosting
- SVC RBF
- Logistic Regression

The final model is selected using cross-validated F1 score. The selected model is then optionally calibrated with isotonic probability calibration, which is enabled by default.

## Training Results

Saved report:

```text
pkl_models/training_report.json
```

Best model:

```text
extra_trees
```

Cross-validation results:

```text
Extra Trees:         F1 0.9546, ROC AUC 0.9909, Accuracy 0.9550
Random Forest:       F1 0.9538, ROC AUC 0.9902, Accuracy 0.9540
Gradient Boosting:   F1 0.9528, ROC AUC 0.9898, Accuracy 0.9533
SVC RBF:             F1 0.9452, ROC AUC 0.9872, Accuracy 0.9460
Logistic Regression: F1 0.9336, ROC AUC 0.9842, Accuracy 0.9343
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

## Why Extra Trees Was Selected

Extra Trees performs well for this project because the model learns from structured numerical features: URL length, entropy, suspicious words, domain flags, text statistics, and other tabular signals.

Tree-based models can learn feature interactions such as:

```text
High URL entropy + suspicious TLD -> higher scam risk
Free hosting + brand mismatch -> higher scam risk
Long hostname + many digits -> higher scam risk
Scam keywords in content -> higher scam risk
```

This makes Extra Trees a better fit than a simple linear model for the current feature set.

## Real-Site Evaluation

Run evaluation against known legitimate sites and fresh scam/phishing feeds:

```powershell
python evaluate_real_sites.py --legit-count 23 --scam-count 20 --timeout 5
```

Latest saved real-site result:

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

The scam URLs are pulled from OpenPhish first, then URLhaus if more samples are needed. Scam page content scraping is disabled by default for safer evaluation; URL features are still extracted.

## Predict Any URL

Predict a single URL:

```powershell
python predict.py --url https://example.com
```

Predict with a custom scraping timeout:

```powershell
python predict.py --url https://example.com --timeout 5
```

Predict multiple URLs from a text file:

```powershell
python predict.py --input urls.txt
```

Predict from a CSV file with a `url` column and save results:

```powershell
python predict.py --input urls.csv --output dataset\predictions.csv
```

Output includes:

```text
URL
Final URL after redirects
Scrape status
Prediction: LEGIT or SCAM
Confidence
Scam probability
Selected feature values
```

## Other Scripts

`build_reference_dataset.py`

Builds a balanced training CSV from the local reference datasets.

`train_model.py`

Cleans the dataset, compares candidate models, saves the best calibrated model, saves the feature list, and writes `training_report.json`.

`evaluate_real_sites.py`

Tests the saved model against legitimate websites and scam/phishing feeds.

`predict.py`

Predicts one URL, a text file of URLs, or a CSV of URLs.

`export_dataset.py`

Exports labeled website records from MySQL into a CSV dataset.

`main.py`

Collects website data, extracts URL/domain/content features, and stores records in MySQL.

`batch_test.py`

Runs a small manually defined batch of URLs and writes a markdown report.

## Important Notes

Predictions should be treated as **risk scores**, not absolute truth. Legitimate websites can sometimes look suspicious, and phishing sites change quickly.

Recommended report wording:

```text
The model performs well on the prepared dataset and current real-site sample
testing, achieving 43/43 correct predictions in the latest saved evaluation.
However, phishing patterns and legitimate website structures vary, so model
outputs should be treated as risk scores rather than absolute truth.
```

