from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Using HealthFact-based classifier
classifier = pipeline("text-classification", model="mbzuai/bioformer-cased-base-finetuned-healthfact")

@app.route('/check', methods=['POST'])
def check_text():
    data = request.get_json()
    text = data.get("text", "")
    result = classifier(text)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)

