# Spam/Ham Message Classifier

A machine learning model that classifies text messages as **SPAM** or **HAM** (legitimate) using text preprocessing and Naive Bayes classification algorithm.

## Project Structure

```
├── spam_classifier.py      # Main model and training code
├── predict.py              # Prediction mode for new messages
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── models/                 # Saved models directory
    └── spam_classifier_model.pkl
```

## Features

### Text Preprocessing
- **Lowercasing**: Converts all text to lowercase for uniformity
- **URL/Email Removal**: Removes URLs and email patterns
- **Special Character Removal**: Strips punctuation and symbols
- **Tokenization**: Breaks text into individual words
- **Stopword Removal**: Removes common English stopwords (the, a, an, etc.)

### Classification Algorithm
- **Algorithm**: Multinomial Naive Bayes
- **Feature Extraction**: TF-IDF Vectorization (max 3000 features)
- **Training/Testing Split**: 80/20 ratio with stratification

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

## Installation

### Requirements
- Python 3.7+
- pip (Python package manager)

### Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   Or manually install:
   ```bash
   pip install scikit-learn pandas numpy
   ```

## Usage

### Method 1: Train and Evaluate (Recommended First Time)

Run the main classifier script:
```bash
python spam_classifier.py
```

**Output includes:**
- Model training confirmation
- Performance metrics (Accuracy, Precision, Recall, F1-Score)
- Confusion matrix
- Sample predictions on test messages
- Saved model file

### Method 2: Interactive Prediction Mode

After training, use the prediction script:
```bash
python predict.py
```

Or specify a custom model path:
```bash
python predict.py models/my_custom_model.pkl
```

**Usage:**
- Enter messages one at a time
- View SPAM/HAM classification with probability scores
- Type `quit` to exit

### Method 3: Programmatic Usage

```python
from spam_classifier import SpamClassifier

# Initialize classifier
classifier = SpamClassifier()

# Load pre-trained model
classifier.load_model('models/spam_classifier_model.pkl')

# Make predictions
results = classifier.predict([
    "You've won a free prize! Click here!",
    "Hey, how are you doing?"
])

for result in results:
    print(f"Message: {result['message']}")
    print(f"Classification: {result['classification']}")
    print(f"Spam Probability: {result['spam_probability']:.1%}")
```

## Model Performance

### Typical Results (with sample data)
- **Accuracy**: 95-100%
- **Precision**: High true positive rate
- **Recall**: Good spam detection rate
- **F1-Score**: Balanced performance

*Note: Actual performance depends on dataset quality and diversity*

## Sample Dataset

The model includes a sample dataset with:
- **Spam messages**: 10 examples (prizes, offers, suspicious links)
- **Ham messages**: 10 examples (everyday conversations, appointments)

To use your own dataset, modify the `create_sample_dataset()` function in `spam_classifier.py` or create a CSV file with columns: `message`, `label` (0=ham, 1=spam).

## File Descriptions

### spam_classifier.py
Main module containing:
- **TextPreprocessor**: Handles text cleaning and tokenization
- **SpamClassifier**: Main classifier with training and prediction
- **create_sample_dataset()**: Generates sample data
- **main()**: Training pipeline and evaluation

Key methods:
- `preprocess()`: Clean text
- `train()`: Train the model
- `predict()`: Classify new messages
- `evaluate()`: Get performance metrics
- `save_model()`: Save trained model
- `load_model()`: Load saved model

### predict.py
Interactive prediction script for classifying new messages after model is trained.

### requirements.txt
Python package dependencies and versions.

## Performance Optimization Tips

1. **Increase training data**: More diverse examples improve generalization
2. **Adjust features**: Modify `max_features` in TfidfVectorizer (default: 3000)
3. **Add more preprocessing**: Implement lemmatization or stemming for better token normalization
4. **Try different algorithms**: Experiment with SVM, Random Forest, or Gradient Boosting
5. **Parameter tuning**: Adjust Naive Bayes parameters for your specific use case

## Common Issues

### Issue: Model not found
**Solution**: Run `python spam_classifier.py` first to train and save the model

### Issue: ImportError for sklearn or pandas
**Solution**: Run `pip install -r requirements.txt`

### Issue: Special characters in paths
**Solution**: Ensure Python can handle the path encoding. Use raw strings or forward slashes.

## Future Enhancements

- [ ] Support for custom datasets (CSV import)
- [ ] Additional classification algorithms comparison
- [ ] REST API for model serving
- [ ] Model versioning and tracking
- [ ] Language support for non-English messages
- [ ] Advanced NLP techniques (word embeddings, deep learning)

## License
MIT License
