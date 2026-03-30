from flask import Flask, request, jsonify
from predict import load_artifacts, predict

app = Flask(__name__)

load_artifacts()

@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.json
    url = data.get("url")

    if not url:
        return jsonify({"error": "No URL provided"}), 400

    result = predict(url, return_json=True)  # modify predict()

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)