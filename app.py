from __future__ import annotations

import json
import os
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

import joblib
from flask import Flask, jsonify, render_template, request

from predict import (
    build_feature_frame,
    normalize_url,
)
from src.content_features import ContentFeatureExtractor
from src.scraper import WebsiteScraper
from src.url_features import URLFeatureExtractor


BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "pkl_models" / "best_scam_detector.pkl"
FEATURES_FILE = BASE_DIR / "pkl_models" / "model_features.pkl"
REPORT_PATH = BASE_DIR / "pkl_models" / "training_report.json"
MAX_HISTORY = 12

app = Flask(__name__)

_state_lock = Lock()
_model = None
_feature_columns: list[str] | None = None
_training_report: dict = {}
_history: deque[dict] = deque(maxlen=MAX_HISTORY)


def _load_artifacts() -> None:
    global _model, _feature_columns, _training_report

    if _model is None:
        if not MODEL_FILE.exists() or not FEATURES_FILE.exists():
            raise FileNotFoundError("Model artifacts not found. Run train_model.py first.")
        _model = joblib.load(MODEL_FILE)
        _feature_columns = joblib.load(FEATURES_FILE)

    if not _training_report and REPORT_PATH.exists():
        _training_report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def _model_summary() -> dict:
    _load_artifacts()
    holdout = _training_report.get("holdout_metrics", {})
    return {
        "ready": True,
        "best_model": _training_report.get("best_model", "saved model"),
        "calibrated": bool(_training_report.get("calibrated", False)),
        "rows": _training_report.get("rows"),
        "feature_count": len(_feature_columns or []),
        "holdout": {
            "accuracy": holdout.get("accuracy"),
            "precision": holdout.get("precision"),
            "recall": holdout.get("recall"),
            "f1": holdout.get("f1"),
            "roc_auc": holdout.get("roc_auc"),
        },
        "features": _feature_columns or [],
    }


def _risk_band(scam_probability: float) -> dict:
    if scam_probability >= 0.75:
        return {"label": "High risk", "tone": "danger"}
    if scam_probability >= 0.5:
        return {"label": "Elevated risk", "tone": "warning"}
    if scam_probability >= 0.3:
        return {"label": "Needs review", "tone": "watch"}
    return {"label": "Low risk", "tone": "safe"}


def _signal_cards(result: dict) -> list[dict]:
    signals = [
        {
            "label": "Suspicious words",
            "value": result.get("suspicious_word_count", 0),
            "active": result.get("suspicious_word_count", 0) > 0,
        },
        {
            "label": "Scam keywords",
            "value": result.get("scam_keyword_count", 0),
            "active": result.get("scam_keyword_count", 0) > 0,
        },
        {
            "label": "Suspicious TLD",
            "value": "Yes" if result.get("suspicious_tld") else "No",
            "active": bool(result.get("suspicious_tld")),
        },
        {
            "label": "URL length",
            "value": result.get("url_length", 0),
            "active": result.get("url_length", 0) >= 80,
        },
        {
            "label": "Page text",
            "value": result.get("text_length", 0),
            "active": result.get("text_length", 0) > 0,
        },
        {
            "label": "Scrape status",
            "value": result.get("status") or "unknown",
            "active": result.get("status") == "success",
        },
    ]
    return signals


def _feature_insights(features: dict, result: dict) -> dict:
    def clamp(value: float) -> float:
        return max(0.0, min(float(value), 1.0))

    lexical = clamp(
        (features.get("url_entropy", 0) / 5.5 * 0.35)
        + (features.get("num_digits", 0) / 18 * 0.2)
        + (features.get("num_hyphen", 0) / 5 * 0.15)
        + (features.get("url_length", 0) / 120 * 0.3)
    )
    host = clamp(
        (features.get("subdomains", 0) / 4 * 0.28)
        + (features.get("suspicious_tld", 0) * 0.26)
        + (features.get("uses_free_hosting", 0) * 0.22)
        + (features.get("has_ip", 0) * 0.24)
    )
    brand = clamp(
        (features.get("brand_in_url", 0) * 0.22)
        + (features.get("brand_domain_mismatch", 0) * 0.5)
        + (features.get("brand_on_free_hosting", 0) * 0.28)
    )
    content = clamp(
        (features.get("scam_keyword_count", 0) / 4 * 0.45)
        + (features.get("suspicious_word_count", 0) / 5 * 0.25)
        + (features.get("caps_ratio", 0) * 0.15)
        + (features.get("exclamation_count", 0) / 4 * 0.15)
    )
    structure = clamp(
        (features.get("num_query_params", 0) / 6 * 0.25)
        + (features.get("path_depth", 0) / 6 * 0.25)
        + (features.get("has_at_symbol", 0) * 0.2)
        + (features.get("tld_in_path", 0) * 0.3)
    )

    bars = [
        {"label": "URL complexity", "value": round(lexical, 3), "detail": "length, entropy, digits"},
        {"label": "Host reputation", "value": round(host, 3), "detail": "TLD, hosting, subdomains"},
        {"label": "Brand mismatch", "value": round(brand, 3), "detail": "brand terms outside official domains"},
        {"label": "Content pressure", "value": round(content, 3), "detail": "scam words and page text"},
        {"label": "Redirect structure", "value": round(structure, 3), "detail": "paths, queries, obfuscation"},
    ]

    strongest = sorted(bars, key=lambda item: item["value"], reverse=True)[:3]
    reliability_notes = []
    if result.get("status") != "success":
        reliability_notes.append("Page content was limited, so URL and hostname signals carry more weight.")
    if features.get("text_length", 0) > 0:
        reliability_notes.append("Content text was available and included in the model input.")
    if features.get("brand_domain_mismatch", 0):
        reliability_notes.append("A known brand appears outside its official registered domain.")
    if not reliability_notes:
        reliability_notes.append("No single high-risk signal dominated this scan.")

    return {
        "radar": bars,
        "top_drivers": strongest,
        "notes": reliability_notes,
    }


def _predict_with_features(
    url: str,
    scraper: WebsiteScraper,
    url_extractor: URLFeatureExtractor,
    content_extractor: ContentFeatureExtractor,
) -> dict:
    scraped = scraper.scrape(url)
    final_url = scraped.get("final_url") or url

    features = {}
    features.update(url_extractor.extract(final_url))
    features.update(
        content_extractor.extract(
            title=scraped.get("title"),
            text=scraped.get("text"),
            html=scraped.get("html"),
        )
    )

    feature_frame = build_feature_frame(_feature_columns or [], features)
    if hasattr(_model, "predict_proba"):
        probabilities = _model.predict_proba(feature_frame)[0]
        scam_probability = float(probabilities[1])
    else:
        scam_probability = float(_model.predict(feature_frame)[0])

    prediction = int(scam_probability >= 0.5)
    prediction_name = "SCAM" if prediction == 1 else "LEGIT"
    confidence = scam_probability if prediction == 1 else 1.0 - scam_probability
    model_features = feature_frame.iloc[0].fillna(0).to_dict()

    return {
        "url": url,
        "final_url": final_url,
        "status": scraped.get("status"),
        "error": scraped.get("error"),
        "prediction": prediction_name,
        "prediction_label": prediction,
        "confidence": round(confidence, 4),
        "scam_probability": round(scam_probability, 4),
        "text_length": features.get("text_length", 0),
        "url_length": features.get("url_length", 0),
        "suspicious_tld": features.get("suspicious_tld", 0),
        "suspicious_word_count": features.get("suspicious_word_count", 0),
        "scam_keyword_count": features.get("scam_keyword_count", 0),
        "model_features": model_features,
    }


def _predict_from_feature_dict(raw_url: str, features: dict, status: str = "url_only") -> dict:
    _load_artifacts()
    url = normalize_url(raw_url)
    feature_frame = build_feature_frame(_feature_columns or [], features)
    if hasattr(_model, "predict_proba"):
        probabilities = _model.predict_proba(feature_frame)[0]
        scam_probability = float(probabilities[1])
    else:
        scam_probability = float(_model.predict(feature_frame)[0])

    prediction = int(scam_probability >= 0.5)
    prediction_name = "SCAM" if prediction == 1 else "LEGIT"
    confidence = scam_probability if prediction == 1 else 1.0 - scam_probability
    model_features = feature_frame.iloc[0].fillna(0).to_dict()

    result = {
        "url": url,
        "final_url": url,
        "status": status,
        "error": None,
        "prediction": prediction_name,
        "prediction_label": prediction,
        "confidence": round(confidence, 4),
        "scam_probability": round(scam_probability, 4),
        "text_length": features.get("text_length", 0),
        "url_length": features.get("url_length", 0),
        "suspicious_tld": features.get("suspicious_tld", 0),
        "suspicious_word_count": features.get("suspicious_word_count", 0),
        "scam_keyword_count": features.get("scam_keyword_count", 0),
        "model_features": model_features,
    }
    result["risk_band"] = _risk_band(scam_probability)
    result["signals"] = _signal_cards(result)
    result["insights"] = _feature_insights(model_features, result)
    return result


def _safe_predict(raw_url: str, timeout: int) -> tuple[dict, int]:
    _load_artifacts()
    url = normalize_url(raw_url)
    if not url:
        return {"error": "Please enter a valid URL."}, 400

    timeout = min(max(int(timeout or 6), 2), 20)
    scraper = WebsiteScraper(timeout=timeout)
    url_extractor = URLFeatureExtractor()
    content_extractor = ContentFeatureExtractor()

    try:
        result = _predict_with_features(
            url=url,
            scraper=scraper,
            url_extractor=url_extractor,
            content_extractor=content_extractor,
        )
    except Exception as exc:
        return {"error": f"Prediction failed: {exc}"}, 500

    result["risk_band"] = _risk_band(float(result.get("scam_probability", 0.0)))
    result["signals"] = _signal_cards(result)
    result["insights"] = _feature_insights(result.get("model_features", {}), result)
    result["checked_at"] = datetime.now(timezone.utc).isoformat()

    with _state_lock:
        _history.appendleft(
            {
                "url": result["url"],
                "prediction": result["prediction"],
                "confidence": result["confidence"],
                "scam_probability": result["scam_probability"],
                "risk_band": result["risk_band"],
                "checked_at": result["checked_at"],
            }
        )

    return result, 200


@app.get("/")
def index():
    return render_template("index.html", model_summary=_model_summary())


@app.get("/api/health")
def health():
    try:
        summary = _model_summary()
    except Exception as exc:
        return jsonify({"ready": False, "error": str(exc)}), 503
    return jsonify(summary)


@app.get("/api/model")
def model_info():
    return jsonify(_model_summary())


@app.get("/api/history")
def history():
    with _state_lock:
        return jsonify(list(_history))


@app.post("/api/predict")
def predict_api():
    payload = request.get_json(silent=True) or {}
    result, status_code = _safe_predict(
        raw_url=str(payload.get("url", "")),
        timeout=int(payload.get("timeout", 6) or 6),
    )
    return jsonify(result), status_code


@app.post("/api/features")
def feature_preview_api():
    payload = request.get_json(silent=True) or {}
    url = normalize_url(str(payload.get("url", "")))
    if not url:
        return jsonify({"error": "Please enter a valid URL."}), 400

    _load_artifacts()
    features = URLFeatureExtractor().extract(url)
    frame = build_feature_frame(_feature_columns or [], features)
    return jsonify(
        {
            "url": url,
            "feature_count": len(_feature_columns or []),
            "features": frame.iloc[0].fillna(0).to_dict(),
        }
    )


@app.post("/api/url-predict")
def url_predict_api():
    payload = request.get_json(silent=True) or {}
    url = normalize_url(str(payload.get("url", "")))
    if not url:
        return jsonify({"error": "Please enter a valid URL."}), 400

    try:
        features = URLFeatureExtractor().extract(url)
        features.update(ContentFeatureExtractor().extract(title="", text="", html=""))
        result = _predict_from_feature_dict(url, features)
    except Exception as exc:
        return jsonify({"error": f"URL-only prediction failed: {exc}"}), 500

    return jsonify(result)


if __name__ == "__main__":
    _load_artifacts()
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
