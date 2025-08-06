import os
import random
import nltk
import string
import spacy
from collections import defaultdict, Counter
import math
from sklearn.preprocessing import normalize
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction import DictVectorizer
from sklearn import metrics


nltk.download('stopwords')
nltk.download('punkt')

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
from nltk.util import ngrams


#from transformers import BertTokenizer, BertForSequenceClassification, pipeline




# reads content of each review
def read_reviews(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()

#opens the directory and copies the content into a dictionary
def process_directory(directory):
    reviews = {}
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            reviews[filename] = read_reviews(file_path)
    return reviews

#copies the directories into 2 dictionaries with labels, combines them, shuffles, splits data into 3 sets
def load_data(pos_dir, neg_dir):
    pos_reviews = [(content, "positive") for _, content in process_directory(pos_dir).items()]
    neg_reviews = [(content, "negative") for _, content in process_directory(neg_dir).items()]
    combined_reviews = pos_reviews + neg_reviews
    random.seed(1)
    random.shuffle(combined_reviews)
    total_reviews = len(combined_reviews)

    train_size = int(0.7 * total_reviews)
    eval_size = int(0.2 * total_reviews)

    train_data = combined_reviews[:train_size]
    eval_data = combined_reviews[train_size:train_size + eval_size]
    test_data = combined_reviews[train_size + eval_size:]

    return train_data, eval_data, test_data

#tokenisation methods applied to text 
def preprocess_text(text, method, nlp):
    stoplist = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())

    if method == 'stem':
        st = LancasterStemmer()
        return [st.stem(word) for word in tokens if word not in stoplist and word not in string.punctuation]
    elif method == 'lemmatise' and nlp:
        doc = nlp(text)
        return [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    elif method == 'whitespace':
        return text.split()
    else:
        raise ValueError("Invalid preprocessing method specified.")

#extracts n grams
def extract_ngrams(tokens, n):
    return [' '.join(gram) for gram in ngrams(tokens, n)]

#extracts noun phrases
def extract_noun_phrases(doc, nlp):
    processed_doc = nlp(doc)
    return [chunk.text for chunk in processed_doc.noun_chunks]

#used to extract both n grams and noun phrases
def extract_features(documents, method, nlp, ngram_range, use_ngrams, use_noun_phrases):
    all_features = []
    for doc in documents:
        tokens = preprocess_text(doc, method, nlp) 
        features = tokens

        if use_ngrams:
            for n in range(ngram_range[0], ngram_range[1] + 1):
                features += extract_ngrams(tokens, n)

        if use_noun_phrases and nlp:
            features += extract_noun_phrases(doc, nlp)

        all_features.append(features)
    return all_features


#cuts off tokens by frequency
def frequency_filter(features, min_freq, max_freq):
    word_counts = Counter(feature for document in features for feature in set(document))
    filtered_vocab = {
        word for word, count in word_counts.items()
        if count >= min_freq and (max_freq is None or count <= max_freq)
    }
    return [
        [feature for feature in document if feature in filtered_vocab]
        for document in features
    ]

class TFIDFVectoriser:
    def __init__(self):
        self.doc_count = 0 # total number of documents
        self.term_doc_freq = Counter() # Counter to track each terms document frequency
        self.vocab = set() # set to store the vocabulary


    def fit(self, corpus):
        self.doc_count = len(corpus) #stores the total number of documents
        for document in corpus:  # iterates over each document in the corpus
            unique_terms = set(document) # gets unique terms in the document
            self.vocab.update(unique_terms) # updates the vocabulary with the unique terms
            self.term_doc_freq.update(unique_terms) # increments the document frequency count for each unique term

    def transform(self, corpus):
        tfidf_vectors = []
        for document in corpus:
            term_freq = Counter(document) # counts frequency of each term in the document
            tfidf_vector = {}  # dictionary to store the TF-IDF scores for terms in the document
            for term in term_freq:
                tf = term_freq[term] / len(document) # computes term frequency
                idf = math.log(self.doc_count / (1 + self.term_doc_freq[term])) # computes inverse document frequency
                tfidf_vector[term] = tf * idf # computes TF-IDF score
            tfidf_vectors.append(tfidf_vector) #appends TF-IDF score to the dictionary 
        return tfidf_vectors

def l2_normalize(matrix):
    vectoriser = DictVectorizer(sparse=True)
    sparse_matrix = vectoriser.fit_transform(matrix)
    return normalize(sparse_matrix, norm='l2')

class NaiveBayesClassifier:
    def __init__(self):
        self.prior = {}  # stores prior probabilities for each class
        self.likelihood = {}  # stores likelihood probabilities for each word given a class
        self.vocab = set()  # vocabulary of unique words
        self.class_word_presence = defaultdict(Counter)  # tracks presence of words in documents for each class
        self.class_doc_counts = defaultdict(int)  # tracks document counts for each class
        self.total_docs = 0  # total number of documents

    def train(self, data, labels):
        self.total_docs = len(data)  # set total number of documents

        # iterate over each document and its corresponding label
        for tokens, label in zip(data, labels):
            self.class_doc_counts[label] += 1  # Increment document count for the class

            # tracks binary presence of words in the document
            for word in set(tokens):  # avoids counting duplicates
                self.class_word_presence[label][word] += 1

            # updates the overall vocabulary
            self.vocab.update(tokens)

        # calculates prior probabilities for each class
        self.prior = { c: self.class_doc_counts[c] / self.total_docs for c in self.class_doc_counts}

        # calculates likelihood probabilities for each word given a class using Laplace smoothing
        self.likelihood = {
            c: {word: (self.class_word_presence[c][word] + 1) / (self.class_doc_counts[c] + 2) for word in self.vocab}
             for c in self.class_word_presence}

    def predict(self, tokens):
        class_scores = {}  # stores log probabilities for each class

        for c in self.prior:
            # start with the log prior probability for the class
            class_scores[c] = math.log(self.prior[c])

            # adds log likelihoods for each word based on presence/absence
            for word in self.vocab:
                if word in tokens:
                    class_scores[c] += math.log(self.likelihood[c].get(word, 1 / (self.class_doc_counts[c] + 2)))
                else:
                    # log of the complement for absent words
                    class_scores[c] += math.log(1 - self.likelihood[c].get(word, 1 / (self.class_doc_counts[c] + 2)))

        return max(class_scores, key=class_scores.get) # returns the class with the highest total log probability

#evaluates the performance of the model
def evaluate_model(model, data, preprocess_method, nlp):
    texts, true_labels = zip(*data)
    if isinstance(texts[0], list):
        processed_texts = texts  # assume features are already tokenized
    else:
        processed_texts = [preprocess_text(text, method=preprocess_method, nlp=nlp) for text in texts]

    predictions = [model.predict(tokens) for tokens in processed_texts]
    print(metrics.classification_report(true_labels, predictions))


if __name__ == "__main__":
    pos_dir = "./pos"
    neg_dir = "./neg"

    min_freq = 2
    max_freq = 10000
    method = 'lemmatise'
    ngram_range = (2, 3)
    use_ngrams = False
    use_noun_phrases = False



    #loads data
    train_data, eval_data, test_data = load_data(pos_dir, neg_dir)

    #preprocess training data
    train_texts, train_labels = zip(*train_data)
    nlp = spacy.load("en_core_web_sm")

    #extract compositional features
    compositional_features = extract_features(train_texts, method, nlp, ngram_range, use_ngrams, use_noun_phrases)

    #filters features by frequency
    filtered_train_features = frequency_filter(compositional_features, min_freq, max_freq)

    # TF-IDF feature extraction
    tfidf_vectoriser = TFIDFVectoriser()
    tfidf_vectoriser.fit(filtered_train_features)
    train_vectors = tfidf_vectoriser.transform(filtered_train_features)

    # runs code with and without l2 normalisation
    for normalisation in ['none', 'l2']:
        print(f"\nEvaluating with {normalisation} normalisation:")

        if normalisation == 'l2':
            b_train_vectors = train_vectors
            train_vectors = l2_normalize(train_vectors)

        # trains Naive Bayes
        nb_model = NaiveBayesClassifier()
        nb_model.train(filtered_train_features, train_labels)

        # processes evaluation data
        eval_texts, eval_labels = zip(*eval_data)
        compositional_eval_features = extract_features(eval_texts, method, nlp, ngram_range, use_ngrams, use_noun_phrases)
        filtered_eval_features = frequency_filter(compositional_eval_features, min_freq, max_freq)
        eval_vectors = tfidf_vectoriser.transform(filtered_eval_features)

        if normalisation == 'l2':
            b_eval_vectors = eval_vectors
            eval_vectors = l2_normalize(eval_vectors)

        evaluate_model(nb_model, list(zip(filtered_eval_features, eval_labels)), method, nlp)

    
print("\nEvaluating Scikit-Learn Naive Bayes:")

# uses a single DictVectorizer for training and evaluation
vectoriser = DictVectorizer(sparse=True)

# transforms  data into a sparse matrix
sklearn_train_vectors = vectoriser.fit_transform(b_train_vectors)
sklearn_eval_vectors = vectoriser.transform(b_eval_vectors)

# applies normalisation to the sparse matrices
if normalisation == 'l2':
    sklearn_train_vectors = normalize(sklearn_train_vectors, norm='l2')
    sklearn_eval_vectors = normalize(sklearn_eval_vectors, norm='l2')

# trains and evaluates the scikit-learn Naive Bayes classifier
sklearn_nb = MultinomialNB()
sklearn_nb.fit(sklearn_train_vectors, train_labels)
predictions = sklearn_nb.predict(sklearn_eval_vectors)

# prints the classification report
print(metrics.classification_report(eval_labels, predictions))

