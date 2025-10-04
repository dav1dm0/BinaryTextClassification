
import os
import random
import json
import joblib
import logging
import nltk
import torch
import numpy as np
from typing import List, Tuple, Dict, Any
from collections import Counter, defaultdict
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn import metrics
from transformers import BertTokenizer, BertModel


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)


class NaiveBayesClassifier(BaseEstimator, ClassifierMixin):
    """
    A custom Naive Bayes classifier compatible with scikit-learn.
    """

    def __init__(self):
        self.prior: Dict[Any, float] = {}
        self.likelihood: Dict[Any, Dict[str, float]] = {}
        self.vocab: set = set()
        self.class_word_presence: defaultdict = defaultdict(Counter)
        self.class_doc_counts: defaultdict = defaultdict(int)
        self.total_docs: int = 0

    def fit(self, X: List[List[str]], y: List[Any]) -> 'NaiveBayesClassifier':
        """
        Trains the Naive Bayes classifier.
        Args:
            X: List of tokenized documents.
            y: List of labels.
        Returns:
            self
        """
        self.total_docs = len(X)
        unique_labels = set(y)
        for tokens, label in zip(X, y):
            self.class_doc_counts[label] += 1
            for word in set(tokens):
                self.class_word_presence[label][word] += 1
            self.vocab.update(tokens)
        self.prior = {
            label: self.class_doc_counts[label] / self.total_docs for label in unique_labels
        }
        self.likelihood = {
            label: {
                word: (
                    self.class_word_presence[label][word] + 1) / (self.class_doc_counts[label] + 2)
                for word in self.vocab
            }
            for label in unique_labels
        }
        return self

    def predict(self, X: List[List[str]]) -> List[Any]:
        """
        Makes predictions on new data.
        Args:
            X: List of tokenized documents.
        Returns:
            List of predicted labels.
        """
        predictions = []
        for tokens in X:
            class_scores = {}
            for label, prior in self.prior.items():
                class_scores[label] = np.log(prior)
                for word in self.vocab:
                    if word in tokens:
                        class_scores[label] += np.log(self.likelihood[label].get(
                            word, 1 / (self.class_doc_counts[label] + 2)))
                    else:
                        class_scores[label] += np.log(1 - self.likelihood[label].get(
                            word, 1 / (self.class_doc_counts[label] + 2)))
            predictions.append(max(class_scores, key=class_scores.get))
        return predictions


def load_config(config_path: str = 'config.json') -> dict:
    """
    Loads the configuration file.
    Args:
        config_path: Path to the configuration file.
    Returns:
        Configuration settings as a dictionary.
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_data(pos_dir: str, neg_dir: str) -> Tuple[List[str], List[str]]:
    """
    Loads and shuffles the data.
    Args:
        pos_dir: Path to the directory with positive reviews.
        neg_dir: Path to the directory with negative reviews.
    Returns:
        Tuple containing the combined reviews and labels.
    """
    pos_reviews = [(read_review(os.path.join(pos_dir, f)), 'positive')
                   for f in os.listdir(pos_dir)]
    neg_reviews = [(read_review(os.path.join(neg_dir, f)), 'negative')
                   for f in os.listdir(neg_dir)]
    combined_reviews = pos_reviews + neg_reviews
    random.shuffle(combined_reviews)
    reviews, labels = zip(*combined_reviews)
    return list(reviews), list(labels)


def read_review(file_path: str) -> str:
    """
    Reads a single review file.
    Args:
        file_path: Path to the review file.
    Returns:
        The content of the review.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().strip()


def get_bert_embeddings(
    texts: List[str],
    bert_model: BertModel,
    bert_tokenizer: BertTokenizer,
    batch_size: int = 16
) -> np.ndarray:
    """
    Generates BERT embeddings for a list of texts.
    Args:
        texts: List of texts.
        model: Pre-trained BERT model.
        tokenizer: BERT tokenizer.
    Returns:
        The BERT embeddings as a numpy array.
    """
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = bert_tokenizer(
            texts, return_tensors='pt',
            truncation=True, padding=True, max_length=512
        )
        with torch.no_grad():
            outputs = bert_model(**inputs)
        all_embeddings.append(outputs.last_hidden_state[:, 0, :].cpu().numpy())
        logging.info("Processed batch %d to %d", i, i + len(batch) - 1)
    return np.vstack(all_embeddings)


def create_pipeline(
    classifier_name: str,
    hyperparameters: dict,
    use_bert_embeddings: bool = False
) -> Pipeline:
    """
    Creates a scikit-learn pipeline.
    Args:
        classifier_name: Name of the classifier.
        hyperparameters: Hyperparameters for the classifier.
        use_bert_embeddings: Whether to use BERT embeddings.
    Returns:
        The scikit-learn pipeline.
    """
    steps = []
    if not use_bert_embeddings:
        steps.append(('tfidf', TfidfVectorizer()))

    if classifier_name == 'naive_bayes':
        steps.append(('clf', MultinomialNB()))
    elif classifier_name == 'logistic_regression':
        steps.append(('clf', LogisticRegression(**hyperparameters)))
    elif classifier_name == 'svm':
        steps.append(('clf', SVC(**hyperparameters)))
    elif classifier_name == 'xgboost':
        steps.append(('clf', XGBClassifier(**hyperparameters)))
    else:
        raise ValueError("Unknown classifier: %s" % classifier_name)
    return Pipeline(steps)


def main() -> None:
    """
    Main function to run the NLP pipeline.
    """
    config = load_config()
    file_paths = config['file_paths']
    model_settings = config['model_settings']
    hyperparameters = config['hyperparameters'][model_settings['classifier']]

    # Load data
    reviews, labels = load_data(file_paths['pos_dir'], file_paths['neg_dir'])

    # Initialize BERT model if needed
    bert_model = None
    bert_tokenizer = None
    if model_settings['use_bert_embeddings']:
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_model = BertModel.from_pretrained('bert-base-uncased')

    # Create pipeline
    pipeline = create_pipeline(
        model_settings['classifier'],
        hyperparameters,
        model_settings['use_bert_embeddings']
    )

    # K-fold cross-validation
    kf = KFold(n_splits=model_settings['k_folds'],
               shuffle=True, random_state=42)
    accuracies = []

    for train_index, test_index in kf.split(reviews):
        x_train = [reviews[i] for i in train_index]
        x_test = [reviews[i] for i in test_index]
        y_train = [labels[i] for i in train_index]
        y_test = [labels[i] for i in test_index]

        if model_settings['use_bert_embeddings']:
            x_train = get_bert_embeddings(
                x_train, bert_model, bert_tokenizer, 100)
            x_test = get_bert_embeddings(
                x_test, bert_model, bert_tokenizer, 100)

        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        accuracy = metrics.accuracy_score(y_test, predictions)
        accuracies.append(accuracy)
        logging.info("Fold accuracy: %s", accuracy)
        logging.info("\n%s", metrics.classification_report(
            y_test, predictions))

    logging.info("\nAverage cross-validation accuracy: %.4f",
                 np.mean(accuracies))

    # Train final model on all data
    if model_settings['use_bert_embeddings']:
        final_x = get_bert_embeddings(reviews, bert_model, bert_tokenizer, 100)
    else:
        final_x = reviews

    pipeline.fit(final_x, labels)

    # Save the trained model
    os.makedirs(os.path.dirname(file_paths['saved_model_path']), exist_ok=True)
    joblib.dump(pipeline, file_paths['saved_model_path'])
    logging.info("Model saved to %s", file_paths['saved_model_path'])


def predict_sentiment(
    text: str,
    model_path: str,
    use_bert_embeddings: bool
) -> str:
    """
    Predicts the sentiment of a single text.
    Args:
        text: The text to analyze.
        model_path: Path to the saved model.
        use_bert_embeddings: Whether to use BERT embeddings.
    Returns:
        The predicted sentiment.
    """
    model = joblib.load(model_path)
    if use_bert_embeddings:
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_model = BertModel.from_pretrained('bert-base-uncased')
        processed_text = get_bert_embeddings(
            [text], bert_model, bert_tokenizer, 1)
    else:
        processed_text = [text]
    prediction = model.predict(processed_text)
    return prediction[0]


if __name__ == '__main__':
    main()
    # Example of how to use the prediction function
    config = load_config()
    model_path = config['file_paths']['saved_model_path']
    use_bert = config['model_settings']['use_bert_embeddings']
    SAMPLE_TEXT = (
        "This movie was absolutely fantastic! The acting was superb and the plot was thrilling."
    )
    sentiment = predict_sentiment(SAMPLE_TEXT, model_path, use_bert)
    logging.info("\nPrediction for sample text: '%s' -> %s",
                 SAMPLE_TEXT, sentiment)
