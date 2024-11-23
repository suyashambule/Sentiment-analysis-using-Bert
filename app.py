from flask import Flask, request, jsonify, render_template
from transformers import TFBertForSequenceClassification, BertTokenizer
import tensorflow as tf

app = Flask(__name__)

# Load the model and tokenizer
model = TFBertForSequenceClassification.from_pretrained("/Users/suyash/Desktop/untitled folder 3/saved_model")
tokenizer = BertTokenizer.from_pretrained("/Users/suyash/Desktop/untitled folder 3/saved_model")

@app.route('/')
def index():
    return render_template('index.html')  # Serve the HTML file

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    data = request.get_json()  # Get the JSON data from the request
    review = data.get('review', '')

    if not review:
        return jsonify({'error': 'Review cannot be empty'}), 400

    # Tokenize the review
    inputs = tokenizer(review, padding=True, truncation=True, return_tensors="tf", max_length=512)

    # Get model predictions
    logits = model(**inputs).logits
    probabilities = tf.nn.softmax(logits, axis=-1)

    # Get the sentiment
    sentiment = 'positive' if probabilities[0][1] > probabilities[0][0] else 'negative'

    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)
