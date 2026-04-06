# 🛡️ ScamShield — Scam Website Detection using ML & NLP

A full-stack machine learning system that detects scam and phishing websites in real time by combining **URL analysis**, **domain intelligence**, **NLP-based content analysis**, and a trained **Random Forest classifier**.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Setup & Installation](#setup--installation)
- [Database Configuration](#database-configuration)
- [Usage](#usage)
- [ML Pipeline](#ml-pipeline)
- [API Endpoints](#api-endpoints)
- [Feature Engineering](#feature-engineering)
- [Model Details](#model-details)
- [Dataset](#dataset)

---

## Overview

ScamShield analyses any website URL and returns a verdict — **SCAM 🚨**, **SUSPICIOUS ⚠️**, or **LEGIT ✅** — along with a confidence score and human-readable risk reasons.

The system works by:
1. Scraping the target website (with SSL fallback and redirect tracking)
2. Extracting three feature groups: URL features, domain features, content/NLP features
3. Running a pre-trained Random Forest model
4. Applying rule-based post-processing (brand spoofing, hosting signals, trusted domains)
5. Serving results via a Flask REST API with a responsive HTML/JS frontend

---

## Project Structure

```
Scam_Website_Detection_ML/
│
├── main.py                      # CLI data collection pipeline
├── build_ml_dataset.py          # Export labelled data from MySQL → CSV
├── backfill_url_features.py     # Backfill URL features for existing records
│
├── train_model.ipynb            # Model training notebook (Random Forest + TF-IDF)
├── final_predict_model.ipynb    # Full prediction pipeline notebook
├── visual.ipynb                 # EDA and visualisation notebook
│
├── src/
│   ├── scraper.py               # WebsiteScraper — fetches & parses HTML
│   ├── url_features.py          # URLFeatureExtractor
│   ├── domain_features.py       # DomainFeatureExtractor (WHOIS + SSL)
│   ├── content_features.py      # ContentFeatureExtractor (NLP / keyword)
│   ├── tfidf_vectorizer.py      # TFIDFFeatureExtractor (scikit-learn wrapper)
│   └── database_mysql.py        # DatabaseManager (MySQL connection pool)
│
├── frontend/
│   ├── app.py                   # Flask web application & prediction API
│   ├── index.html               # Frontend UI (v1)
│   └── index1.html              # Frontend UI (v2 — served by default)
│
├── pkl_models/
│   ├── random_forest.pkl        # Trained Random Forest model
│   ├── scaler.pkl               # StandardScaler for numeric features
│   └── feature_columns.pkl      # Ordered feature column list
│
└── dataset/
    ├── ml_project_dataset.csv        # Full labelled dataset
    ├── ml_project_dataset_500.csv    # Sampled 500-row dataset
    └── Scam_website_dataset.csv      # Raw scam URL dataset
```

---

## Features

### 🔗 URL Features (19 features)
| Feature | Description |
|---|---|
| `url_length` | Total character length of the URL |
| `num_dots` | Number of `.` characters |
| `num_hyphen` | Number of `-` characters |
| `num_slashes` | Number of `/` characters |
| `num_digits` | Count of numeric digits in URL |
| `https` | 1 if scheme is HTTPS, else 0 |
| `subdomains` | Number of subdomain levels |
| `has_at_symbol` | Presence of `@` in URL |
| `has_ip` | IP address used instead of domain name |
| `suspicious_tld` | TLD is in known-abused list (`.tk`, `.xyz`, etc.) |
| `brand_in_url` | Brand name (PayPal, Google, etc.) in non-brand domain |
| `is_shortened` | URL is from a shortening service (bit.ly, tinyurl, etc.) |
| `suspicious_word_count` | Count of phishing keywords in URL |
| `path_depth` | Depth of the URL path |
| `num_query_params` | Number of query parameters |

### 🌐 Domain Features (7 features)
| Feature | Description |
|---|---|
| `domain_age_days` | Days since domain registration (via WHOIS) |
| `domain_expiry_days` | Days until domain expiry |
| `is_new_domain` | 1 if domain is less than 180 days old |
| `short_expiry_domain` | 1 if domain expires within 180 days |
| `has_ssl` | SSL certificate present |
| `ssl_valid` | SSL certificate is valid |
| `ssl_days_remaining` | Days until SSL certificate expiry |

### 📄 Content / NLP Features (9 features)
| Feature | Description |
|---|---|
| `text_length` | Character length of page body text |
| `token_count` | Number of meaningful tokens (stopwords removed) |
| `scam_keyword_count` | Count of scam/phishing keywords matched |
| `scam_keyword_density` | Ratio of scam keywords to total tokens |
| `has_form` | HTML `<form>` element present |
| `has_iframe` | HTML `<iframe>` element present |
| `exclamation_count` | Number of `!` characters in body text |
| `caps_ratio` | Ratio of uppercase letters in body text |
| `avg_word_length` | Average word length after tokenisation |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12 |
| ML | scikit-learn (Random Forest, TF-IDF) |
| NLP | NLTK, scikit-learn TfidfVectorizer |
| Web Scraping | requests, BeautifulSoup4, lxml, certifi |
| Domain Intel | tldextract, system `whois`, Python `ssl` + `socket` |
| Backend API | Flask, flask-cors |
| Database | MySQL (mysql-connector-python, connection pooling) |
| Model Serialisation | joblib |
| Data Processing | pandas, numpy |
| Visualisation | matplotlib, seaborn (in notebooks) |
| Notebooks | Jupyter |

---

## Setup & Installation

### Prerequisites
- Python 3.12+
- MySQL 8.0+ (running locally)
- `whois` command-line utility installed on the OS

### 1. Clone the repository
```bash
git clone https://github.com/your-username/Scam_Website_Detection_ML.git
cd Scam_Website_Detection_ML
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download NLTK data
```python
python -c "import nltk; nltk.download('stopwords')"
```

---

## Database Configuration

The project uses a local MySQL database. Update credentials in `src/database_mysql.py` if needed (defaults shown below):

```python
host     = "localhost"
user     = "root"
password = ""
database = "scam_website_detection"
```

Create the database and `websites` table before first run:

```sql
CREATE DATABASE scam_website_detection;
USE scam_website_detection;

CREATE TABLE websites (
    id INT AUTO_INCREMENT PRIMARY KEY,
    url VARCHAR(2048) NOT NULL UNIQUE,
    title TEXT,
    meta_description TEXT,
    text_content LONGTEXT,
    final_url VARCHAR(2048),
    status VARCHAR(20),
    error_message TEXT,
    redirect_count INT DEFAULT 0,
    scraped_at DATETIME,
    label TINYINT,

    -- URL features
    url_length INT, num_dots INT, num_hyphen INT, num_slashes INT,
    num_underscores INT, num_percent INT, num_digits INT,
    https TINYINT, subdomains INT, has_at_symbol TINYINT,
    has_double_slash TINYINT, has_ip TINYINT, num_query_params INT,
    has_query TINYINT, path_depth INT, suspicious_tld TINYINT,
    brand_in_url TINYINT, is_shortened TINYINT, suspicious_word_count INT,

    -- Domain features
    domain_age_days INT, domain_expiry_days INT, registrar VARCHAR(255),
    has_ssl TINYINT, ssl_valid TINYINT, ssl_days_remaining INT,
    is_new_domain TINYINT, short_expiry_domain TINYINT,

    -- Content features
    text_length INT, token_count INT, scam_keyword_count INT,
    scam_keyword_density FLOAT, has_form TINYINT, has_iframe TINYINT,
    exclamation_count INT, caps_ratio FLOAT, avg_word_length FLOAT
);
```

---

## Usage

### Run the Web Application (Recommended)

```bash
cd frontend
python app.py
```

Then open your browser at: **http://localhost:5000**

Enter any URL to get an instant scam verdict with feature breakdown.

---

### Collect & Label Data via CLI

```bash
# Scrape and label a single URL
python main.py --url https://example.com --label 0   # 0 = legit
python main.py --url https://suspicious-site.tk --label 1  # 1 = scam

# Interactive mode (prompts for URL and label)
python main.py
```

### Export Dataset from Database

```bash
python build_ml_dataset.py
# Output: dataset/ml_project_dataset.csv
```

### Train the Model

Open and run `train_model.ipynb` in Jupyter:
```bash
jupyter notebook train_model.ipynb
```

---

## ML Pipeline

```
URL Input
   │
   ├──► URLFeatureExtractor       → 19 numeric features
   ├──► DomainFeatureExtractor    → 7 numeric features (WHOIS + SSL)
   ├──► WebsiteScraper            → HTML / text content
   │         └──► ContentFeatureExtractor → 9 NLP features
   │
   ▼
Feature Vector (35 features)
   │
   ├──► StandardScaler (normalisation)
   ▼
Random Forest Classifier
   │
   ▼
Base ML Probability
   │
   ├──► Rule-based post-processing
   │       (brand spoofing, trusted domains, hosting signals, path keywords)
   ▼
Final Confidence Score → LEGIT / SUSPICIOUS / SCAM
```

---

## API Endpoints

All endpoints are served by `frontend/app.py` on port **5000**.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the frontend UI |
| `GET` | `/api/health` | Health check |
| `POST` | `/api/predict` | Predict scam probability for a URL |
| `GET` | `/api/history` | Returns last 20 predictions (in-memory) |

### POST `/api/predict`

**Request body:**
```json
{ "url": "https://example.com" }
```

**Response:**
```json
{
  "label": "LEGIT",
  "label_emoji": "✅ LEGIT",
  "severity": "low",
  "confidence": 0.23,
  "base_ml_proba": 0.18,
  "scam_score": 0.05,
  "legit_boost": 0.0,
  "domain": "example.com",
  "final_url": "https://www.example.com",
  "page_title": "Example Domain",
  "scrape_status": "success",
  "redirect_count": 1,
  "reasons": [],
  "url_features": { ... },
  "domain_features": { ... },
  "content_features": { ... },
  "timestamp": "2026-04-06T10:30:00"
}
```

---

## Model Details

- **Algorithm:** Random Forest Classifier (scikit-learn)
- **Input features:** 35 engineered features (URL + domain + content)
- **Post-processing rules:** 15+ heuristic rules applied on top of base ML probability
- **Verdicts:**
  - `LEGIT` — final confidence < 0.60
  - `SUSPICIOUS` — final confidence 0.60–0.79
  - `SCAM` — final confidence ≥ 0.80
- **Serialised artifacts:** `pkl_models/random_forest.pkl`, `pkl_models/scaler.pkl`, `pkl_models/feature_columns.pkl`

---

## Dataset

Three CSV files are included under `dataset/`:

| File | Description |
|---|---|
| `ml_project_dataset.csv` | Full labelled dataset exported from MySQL |
| `ml_project_dataset_500.csv` | Sampled 500-record subset for quick experiments |
| `Scam_website_dataset.csv` | Raw scam URL collection used for initial labelling |

Labels: `0` = Legitimate, `1` = Scam

---

## Notes

- WHOIS lookups require the system `whois` tool to be installed (`sudo apt install whois` on Linux).
- Domain feature extraction makes live network calls (WHOIS + SSL); set appropriate timeouts in production.
- The `registrar` field from WHOIS is stored in the database but not used as an ML feature.
- TF-IDF features are used during training but the saved Random Forest model uses the 35 engineered features only (no TF-IDF at inference time in the Flask app).
