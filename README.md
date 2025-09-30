# Sentiment Analysis using BERT with Web Interface

This project implements **BERT (Bidirectional Encoder Representations from Transformers)** for sentiment analysis on movie reviews from the IMDB dataset, complete with a **Flask web application** for real-time sentiment prediction. The model leverages state-of-the-art transformer architecture to achieve high accuracy in binary sentiment classification.

## 🎯 Project Overview

This comprehensive sentiment analysis solution demonstrates:
- **BERT Integration**: Fine-tuned BERT-base-uncased model for sequence classification
- **Web Deployment**: Interactive Flask application with responsive HTML interface
- **Real-time Predictions**: Live sentiment analysis through RESTful API
- **Production Ready**: Complete deployment pipeline from training to serving
- **High Performance**: 92.77% accuracy achieved with transformer architecture

## 🏗️ Architecture Overview

### Model Architecture
```
BERT-base-uncased (Pre-trained)
├── Transformer Layers: 12 layers
├── Hidden Size: 768
├── Attention Heads: 12
├── Parameters: ~110M
└── Classification Head: Dense(2) + Dropout
```

### Web Application Stack
```
Frontend: HTML + CSS + JavaScript
├── Responsive Design
├── Real-time Form Submission
└── Dynamic Result Display

Backend: Flask REST API
├── BERT Model Loading
├── Text Tokenization
└── Sentiment Prediction
```

## 📊 Dataset

**IMDB Movie Reviews Dataset**
- **Source**: Hugging Face Datasets - 'imdb'
- **Size**: 50,000 movie reviews (25K train, 25K test)
- **Classes**: Binary classification (Positive: 1, Negative: 0)
- **Format**: Text reviews with sentiment labels
- **Quality**: Professional movie reviews with rich vocabulary

## 🔧 Key Features

### Model Features
- **Pre-trained BERT**: Leverages bert-base-uncased with 110M parameters
- **Fine-tuning**: Specialized training on IMDB movie review data
- **Advanced Tokenization**: WordPiece tokenization with 512 max length
- **Optimization**: AdamW optimizer with learning rate scheduling
- **Efficiency**: Batch processing with TensorFlow integration

### Web Application Features
- **Interactive Interface**: Clean, user-friendly web form
- **Real-time Analysis**: Instant sentiment prediction
- **Visual Feedback**: Color-coded results (green/red)
- **Error Handling**: Robust input validation and error messages
- **Responsive Design**: Mobile-friendly interface

## 📋 Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Main Dependencies:
- `tensorflow>=2.10.0` - Deep learning framework
- `transformers>=4.20.0` - Hugging Face transformers library
- `datasets>=2.0.0` - Dataset loading and processing
- `flask>=2.2.0` - Web framework for deployment
- `pandas>=1.3.0` - Data manipulation
- `numpy>=1.21.0` - Numerical computing

## 🚀 Getting Started

### 1. Clone and Setup
```bash
cd Sentiment-analysis-using-Bert
pip install -r requirements.txt
```

### 2. Train the Model (Optional)
```python
# Run the Jupyter notebook to train from scratch
jupyter notebook "SentimentalanalysiswithBert.ipynb"
```

### 3. Run the Web Application
```bash
# Start the Flask server
python app.py

# Open your browser and navigate to:
# http://localhost:5000
```

### 4. Using the API
```python
# Direct API call
import requests

response = requests.post('http://localhost:5000/analyze', 
                        json={'review': 'This movie is amazing!'})
print(response.json())  # {'sentiment': 'positive'}
```

## 📈 Model Performance

The BERT model demonstrates excellent performance:

| Metric | Value |
|--------|-------|
| **Training Accuracy** | 92.77% |
| **Training Loss** | 0.2001 |
| **Training Time** | ~497 seconds (1000 batches) |
| **Model Size** | 417.65 MB |
| **Parameters** | 109,483,778 |

### Training Configuration:
- **Learning Rate**: 2e-5
- **Batch Size**: 4
- **Max Sequence Length**: 512
- **Optimizer**: AdamW with linear schedule
- **Training Steps**: 1000 (subset for demo)

## 🔬 Technical Implementation

### Model Training
```python
# Load and preprocess dataset
dataset = load_dataset('imdb')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize data
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding=True, max_length=512)

# Fine-tune BERT model
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')
model.compile(optimizer=optimizer, metrics=['accuracy'])
```

### Web Application
```python
# Flask API endpoint
@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    data = request.get_json()
    review = data.get('review', '')
    
    # Tokenize input
    inputs = tokenizer(review, padding=True, truncation=True, 
                      return_tensors="tf", max_length=512)
    
    # Get predictions
    logits = model(**inputs).logits
    probabilities = tf.nn.softmax(logits, axis=-1)
    
    # Return sentiment
    sentiment = 'positive' if probabilities[0][1] > probabilities[0][0] else 'negative'
    return jsonify({'sentiment': sentiment})
```

## 📁 Project Structure

```
Sentiment-analysis-using-Bert/
├── SentimentalanalysiswithBert.ipynb  # Main training notebook
├── app.py                            # Flask web application
├── index.html                        # Frontend interface
├── README.md                         # Project documentation
├── requirements.txt                  # Dependencies list
└── saved_model/                      # Trained model directory
    ├── config.json                   # Model configuration
    ├── tf_model.h5                   # Model weights
    ├── tokenizer_config.json         # Tokenizer settings
    └── vocab.txt                     # Vocabulary file
```

## 🌐 Web Interface

### Features:
- **Input Form**: Large text area for movie review input
- **Submit Button**: Trigger sentiment analysis
- **Results Display**: Color-coded sentiment output
- **Loading State**: Visual feedback during processing
- **Error Handling**: User-friendly error messages

### Usage:
1. Enter a movie review in the text area
2. Click "Analyze Sentiment" button
3. View the predicted sentiment (Positive/Negative)
4. Results are color-coded for quick interpretation

## 🎓 Learning Objectives

This project demonstrates:
1. **Transformer Architecture**: Understanding BERT and attention mechanisms
2. **Fine-tuning**: Adapting pre-trained models for specific tasks
3. **Web Deployment**: Creating production-ready ML applications
4. **API Design**: Building RESTful endpoints for ML models
5. **Full-stack Development**: Integrating ML models with web interfaces
6. **Model Serving**: Production deployment strategies

## 🔍 Key Technical Insights

- **BERT Effectiveness**: Transformer architecture excels at understanding context
- **Pre-training Benefits**: Transfer learning significantly reduces training requirements
- **Tokenization Impact**: WordPiece tokenization handles out-of-vocabulary words
- **Batch Processing**: Efficient inference with TensorFlow batch operations
- **Web Integration**: Seamless model serving through Flask API

## 🚀 Future Enhancements

Potential improvements and extensions:
- [ ] **Model Upgrades**: Implement RoBERTa, DistilBERT, or larger BERT variants
- [ ] **Multi-class Sentiment**: Extend to fine-grained sentiment levels (1-5 stars)
- [ ] **Real-time Streaming**: WebSocket integration for live predictions
- [ ] **Model Comparison**: A/B testing interface with multiple models
- [ ] **Performance Optimization**: Model quantization and TensorRT optimization
- [ ] **Cloud Deployment**: Deploy on AWS, GCP, or Azure
- [ ] **Batch Processing**: Upload CSV files for bulk sentiment analysis
- [ ] **Visualization Dashboard**: Charts and analytics for sentiment trends

## 📊 Deployment Options

### Local Development
```bash
python app.py  # Development server
```

### Production Deployment
```bash
# Using Gunicorn
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# Using Docker
docker build -t sentiment-app .
docker run -p 5000:5000 sentiment-app
```

### Cloud Platforms
- **Heroku**: Easy deployment with git integration
- **AWS EC2/ECS**: Scalable container deployment
- **Google Cloud Run**: Serverless container platform
- **Azure Container Instances**: Managed container service

## 🛠️ Technical Stack

**Machine Learning:**
- TensorFlow 2.x with Keras API
- Hugging Face Transformers
- BERT-base-uncased model
- Datasets library for data loading

**Web Development:**
- Flask web framework
- HTML5 + CSS3 + JavaScript
- RESTful API design
- JSON data exchange

**Development Tools:**
- Jupyter Notebook for experimentation
- Python 3.8+
- Git version control

## 🤝 Contributing

Feel free to contribute to this project by:
1. Reporting bugs or issues
2. Suggesting new features or improvements
3. Submitting pull requests
4. Improving documentation
5. Adding new model architectures
6. Enhancing the web interface

*This project serves as a comprehensive example of modern NLP applications, combining state-of-the-art transformer models with practical web deployment for real-world sentiment analysis tasks.*
