"""
Spam/Ham Message Classifier using Naive Bayes with Real Dataset
This module implements text preprocessing and classification for spam/ham detection.
Dataset: D:\Downloads\archive\spam.csv (5572 messages)
"""

import re
import string
import numpy as np
import csv
import os
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle


class TextPreprocessor:
    """Handles text preprocessing for spam/ham classification."""
    
    def __init__(self):
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'is', 'was', 'are', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'
        }
    
    def clean_text(self, text):
        """Convert text to lowercase and remove special characters."""
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|email', '', text)
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return text
    
    def tokenize(self, text):
        """Split text into words."""
        return text.split()
    
    def remove_stopwords(self, tokens):
        """Remove common stopwords."""
        return [token for token in tokens if token not in self.stop_words]
    
    def preprocess(self, text):
        """Apply full preprocessing pipeline."""
        text = self.clean_text(text)
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        return ' '.join(tokens)


class SpamClassifier:
    """Spam/Ham classifier using Naive Bayes."""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
        self.classifier = MultinomialNB()
        self.is_trained = False
    
    def prepare_data(self, messages, labels):
        """Preprocess messages and prepare data for training."""
        processed_messages = [self.preprocessor.preprocess(msg) for msg in messages]
        X = self.vectorizer.fit_transform(processed_messages)
        y = np.array(labels)
        return X, y
    
    def train(self, X_train, y_train):
        """Train the classifier."""
        self.classifier.fit(X_train, y_train)
        self.is_trained = True
        print("✓ Model trained successfully")
    
    def predict(self, messages):
        """Predict spam/ham for new messages."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if isinstance(messages, str):
            messages = [messages]
        
        processed = [self.preprocessor.preprocess(msg) for msg in messages]
        X = self.vectorizer.transform(processed)
        predictions = self.classifier.predict(X)
        probabilities = self.classifier.predict_proba(X)
        
        results = []
        for msg, pred, prob in zip(messages, predictions, probabilities):
            results.append({
                'message': msg,
                'classification': 'SPAM' if pred == 1 else 'HAM',
                'spam_probability': prob[1],
                'ham_probability': prob[0]
            })
        return results
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance."""
        predictions = self.classifier.predict(X_test)
        
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        cm = confusion_matrix(y_test, predictions)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm
        }
        return metrics, predictions
    
    def save_model(self, filepath):
        """Save trained model and vectorizer."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        model_data = {
            'classifier': self.classifier,
            'vectorizer': self.vectorizer,
            'preprocessor': self.preprocessor
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"✓ Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model and vectorizer."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        self.classifier = model_data['classifier']
        self.vectorizer = model_data['vectorizer']
        self.preprocessor = model_data['preprocessor']
        self.is_trained = True
        print(f"✓ Model loaded from {filepath}")


def create_sample_dataset():
    """Load dataset from CSV file."""
    import csv
    import os
    
    # Path to the spam.csv dataset
    dataset_path = r"D:\Downloads\archive\spam.csv"
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    messages = []
    labels = []
    
    try:
        with open(dataset_path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header row (v1, v2, ...)
            
            for row in reader:
                if len(row) >= 2:
                    label = row[0].strip().lower()
                    message = row[1].strip()
                    
                    if message:  # Only include non-empty messages
                        messages.append(message)
                        labels.append(1 if label == 'spam' else 0)
    except Exception as e:
        print(f"Error reading dataset: {e}")
        raise
    
    print(f"✓ Loaded {len(messages)} messages from {dataset_path}")
    return messages, labels


def main():
    """Main function to train and evaluate the spam classifier."""
    
    print("=" * 60)
    print("SPAM/HAM CLASSIFICATION MODEL")
    print("=" * 60)
    
    print("\n📊 Loading dataset from CSV...")
    messages, labels = create_sample_dataset()
    print(f"   Total messages: {len(messages)}")
    print(f"   Spam messages: {sum(labels)}")
    print(f"   Ham messages: {len(labels) - sum(labels)}")
    
    print("\n🔄 Splitting data into train/test sets...")
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        messages, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print("\n🏋️ Initializing and training classifier...")
    classifier = SpamClassifier()
    X_train, _ = classifier.prepare_data(X_train_text, y_train)
    classifier.train(X_train, y_train)
    
    print("\n📈 Evaluating model performance...")
    X_test, _ = classifier.prepare_data(X_test_text, y_test)
    metrics, predictions = classifier.evaluate(X_test, y_test)
    
    print("\n" + "=" * 60)
    print("MODEL EVALUATION METRICS")
    print("=" * 60)
    print(f"Accuracy:  {metrics['accuracy']:.2%}")
    print(f"Precision: {metrics['precision']:.2%}")
    print(f"Recall:    {metrics['recall']:.2%}")
    print(f"F1-Score:  {metrics['f1_score']:.2%}")
    print("\nConfusion Matrix:")
    print(f"  True Negatives:  {metrics['confusion_matrix'][0][0]}")
    print(f"  False Positives: {metrics['confusion_matrix'][0][1]}")
    print(f"  False Negatives: {metrics['confusion_matrix'][1][0]}")
    print(f"  True Positives:  {metrics['confusion_matrix'][1][1]}")
    
    print("\n" + "=" * 60)
    print("SAMPLE PREDICTIONS")
    print("=" * 60)
    
    test_messages = [
        "You've won a free laptop! Click here now!",
        "Hey, are you available for coffee tomorrow?",
        "URGENT: Claim your prize money immediately!",
        "Let's meet at the office at 3 PM."
    ]
    
    predictions_result = classifier.predict(test_messages)
    for i, result in enumerate(predictions_result, 1):
        print(f"\n{i}. Message: {result['message']}")
        print(f"   Classification: {result['classification']}")
        print(f"   Spam Probability: {result['spam_probability']:.2%}")
    
    print("\n" + "=" * 60)
    print(" Saving model...")
    model_path = 'models/spam_classifier_model.pkl'
    classifier.save_model(model_path)
    
    print("\n Model training and evaluation complete!")


if __name__ == '__main__':
    main()
