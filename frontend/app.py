import os, sys, json
from datetime import datetime
from urllib.parse import urlparse

_FRONTEND = os.path.dirname(os.path.abspath(__file__))   
_ROOT     = os.path.dirname(_FRONTEND)                   
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import joblib
from flask import Flask, request, jsonify, send_from_directory
try:
    from flask_cors import CORS; HAS_CORS = True
except ImportError:
    HAS_CORS = False

from src.scraper          import WebsiteScraper
from src.url_features     import URLFeatureExtractor
from src.domain_features  import DomainFeatureExtractor
from src.content_features import ContentFeatureExtractor

# load artifacts
MODEL_PATH      = os.path.join(_ROOT, 'pkl_models')
model           = joblib.load(os.path.join(MODEL_PATH, 'random_forest.pkl'))
feature_columns = joblib.load(os.path.join(MODEL_PATH, 'feature_columns.pkl'))
print("[INFO] Random Forest model loaded")
print(f"[INFO] Feature columns: {len(feature_columns)}")

# extractors
scraper     = WebsiteScraper()
url_ext     = URLFeatureExtractor()
domain_ext  = DomainFeatureExtractor()
content_ext = ContentFeatureExtractor()

# in-memory history
_history = []

#  PREDICTION LOGIC

def build_vector(features: dict):
    import pandas as pd
    df = pd.DataFrame([features])
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    return df[feature_columns]


def run_predict(url: str) -> dict:
    """
    Exact logic from final_predict_model.ipynb – returns a dict for the API.
    """
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url

    # scrape
    data      = scraper.scrape(url)
    final_url = data.get('final_url', url)
    domain    = urlparse(final_url).netloc.lower()

    # features
    features = {}
    features.update(url_ext.extract(final_url))
    features.update(domain_ext.extract(final_url))
    features.update(content_ext.extract(
        data.get('title'),
        data.get('text'),
        data.get('html')
    ))

    X          = build_vector(features)
    base_proba = float(model.predict_proba(X)[0][1])

    scam_score  = 0.0
    legit_boost = 0.0
    reasons     = []

    # trust signals
    trusted_domains = ['.edu', '.gov', '.ac.', '.org', '.mx']
    is_trusted = any(td in domain for td in trusted_domains)

    if is_trusted:
        legit_boost += 0.40
        reasons.append("Trusted domain")

    age = features.get('domain_age_days', 0)
    if age and age > 1000:
        legit_boost += 0.05
        reasons.append("Old domain")

    # shortened url
    is_short_url = False
    if features.get('is_shortened') == 1:
        is_short_url = True
        scam_score  += 0.25
        reasons.append("Shortened URL")

    # hosting domain
    hosting_domains = [
        "wpenginepowered", "herokuapp", "firebaseapp",
        "github.io", "netlify", "vercel", "000webhost"
    ]
    if any(h in domain for h in hosting_domains):
        scam_score += 0.30
        reasons.append("Suspicious hosting domain")

    # scam signals
    if features.get('num_hyphen', 0) >= 5:
        scam_score += 0.15
        reasons.append("Excessive hyphens")

    if features.get('subdomains', 0) >= 3:
        scam_score += 0.15
        reasons.append("Too many subdomains")

    if features.get('suspicious_tld') == 1:
        scam_score += 0.20
        reasons.append("Suspicious TLD")

    if any(h in domain for h in ['weebly', 'blogspot', 'wixsite', 'wordpress']):
        scam_score += 0.25
        reasons.append("Free hosting domain")

    if features.get('https') == 0:
        scam_score += 0.05
        reasons.append("No HTTPS")

    if features.get('url_length', 0) > 80:
        scam_score += 0.05
        reasons.append("Very long URL")

    # path-based phishing
    suspicious_paths = ["login", "verify", "account", "update", "secure", "wp-admin"]
    if any(p in final_url.lower() for p in suspicious_paths):
        scam_score += 0.20
        reasons.append("Suspicious path keywords")

    # content signals
    if features.get('scam_keyword_density', 0) > 0.05:
        scam_score += 0.15
        reasons.append("High scam keyword density")

    if features.get('has_form') == 1 and not is_trusted:
        scam_score += 0.05
        reasons.append("Form present")

    if features.get('has_iframe') == 1 and not is_trusted:
        scam_score += 0.05
        reasons.append("Iframe detected")

    # final probability
    final_proba = (0.85 * base_proba) + scam_score - legit_boost
    final_proba = max(0.0, min(1.0, final_proba))

    if is_trusted:
        final_proba = min(final_proba, 0.45)

    # decision
    if is_short_url:
        label       = "SUSPICIOUS"
        label_emoji = "⚠️ SUSPICIOUS (Shortened URL)"
        severity    = "medium"
        final_proba = max(final_proba, 0.65)

    elif final_proba >= 0.80:
        label       = "SCAM"
        label_emoji = "🚨 SCAM (HIGH RISK)"
        severity    = "high"

    elif final_proba >= 0.60:
        label       = "SUSPICIOUS"
        label_emoji = "⚠️ SUSPICIOUS"
        severity    = "medium"

    else:
        label       = "LEGIT"
        label_emoji = "✅ LEGIT"
        severity    = "low"

    # safe cast helpers
    def _i(k):
        v = features.get(k)
        try:   return int(v)   if v is not None else None
        except: return None

    def _f(k):
        v = features.get(k)
        try:   return round(float(v), 4) if v is not None else None
        except: return None

    return {
        # verdict
        "label":          label,
        "label_emoji":    label_emoji,
        "severity":       severity,
        "confidence":     round(final_proba, 4),
        "base_ml_proba":  round(base_proba,  4),
        "scam_score":     round(scam_score,  4),
        "legit_boost":    round(legit_boost, 4),

        # page info
        "domain":         domain,
        "final_url":      final_url,
        "page_title":     data.get("title", ""),
        "scrape_status":  data.get("status", "unknown"),
        "redirect_count": int(data.get("redirect_count", 0) or 0),

        # reasons
        "reasons": reasons,

        # url features
        "url_features": {
            "url_length":            _i("url_length"),
            "num_dots":              _i("num_dots"),
            "num_hyphen":            _i("num_hyphen"),
            "num_slashes":           _i("num_slashes"),
            "num_digits":            _i("num_digits"),
            "num_query_params":      _i("num_query_params"),
            "path_depth":            _i("path_depth"),
            "subdomains":            _i("subdomains"),
            "https":                 _i("https"),
            "has_at_symbol":         _i("has_at_symbol"),
            "has_ip":                _i("has_ip"),
            "suspicious_tld":        _i("suspicious_tld"),
            "brand_in_url":          _i("brand_in_url"),
            "is_shortened":          _i("is_shortened"),
            "suspicious_word_count": _i("suspicious_word_count"),
        },

        # domain features
        "domain_features": {
            "domain_age_days":     _i("domain_age_days"),
            "domain_expiry_days":  _i("domain_expiry_days"),
            "has_ssl":             _i("has_ssl"),
            "ssl_valid":           _i("ssl_valid"),
            "ssl_days_remaining":  _i("ssl_days_remaining"),
            "is_new_domain":       _i("is_new_domain"),
            "short_expiry_domain": _i("short_expiry_domain"),
        },

        # content features
        "content_features": {
            "text_length":          _i("text_length"),
            "token_count":          _i("token_count"),
            "scam_keyword_count":   _i("scam_keyword_count"),
            "scam_keyword_density": _f("scam_keyword_density"),
            "has_form":             _i("has_form"),
            "has_iframe":           _i("has_iframe"),
            "exclamation_count":    _i("exclamation_count"),
            "caps_ratio":           _f("caps_ratio"),
            "avg_word_length":      _f("avg_word_length"),
        },
    }

#  FLASK APP
app = Flask(__name__, static_folder=_FRONTEND)
if HAS_CORS:
    CORS(app)


@app.route("/", methods=["GET"])
def index():
    return send_from_directory(_FRONTEND, "index.html")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "ScamShield ML", "version": "1.0"})


@app.route("/api/predict", methods=["POST"])
def predict_endpoint():
    body = request.get_json(force=True)
    url  = (body.get("url") or "").strip()
    if not url:
        return jsonify({"error": "No URL provided"}), 400
    try:
        result = run_predict(url)
        result["timestamp"] = datetime.now().isoformat()

        _history.insert(0, {
            "url":        url,
            "label":      result["label"],
            "confidence": result["confidence"],
            "severity":   result["severity"],
            "timestamp":  result["timestamp"],
        })
        if len(_history) > 20:
            _history.pop()

        return jsonify(result)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/history", methods=["GET"])
def history():
    return jsonify({"history": _history})


if __name__ == "__main__":
    print(f"\n🌐  ScamShield ML → http://localhost:5000\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
