import nltk
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize

text = "NLTK is a powerful library for natural language processing. It helps in tokenization."
words = word_tokenize(text)
sentences = sent_tokenize(text)

print(words)
print(sentences)


stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

word = "running"
stemmed_word = stemmer.stem(word)
lemmatized_word = lemmatizer.lemmatize(word)

print(stemmed_word)
print(lemmatized_word)


text = "NLTK is a powerful library for natural language processing."

words = word_tokenize(text)
tags = pos_tag(words)

print(tags)

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


# Load movie reviews corpus
documents = [(list(movie_reviews.words(fileid)), category) for category in movie_reviews.categories() for fileid in movie_reviews.fileids(category)]

# Define features
all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
word_features = list(all_words.keys())[:2000]

# Define feature extraction function
def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features[f'contains({word})'] = (word in document_words)
    return features

# Extract features and train a classifier
featuresets = [(document_features(d), c) for (d, c) in documents]
train_set, test_set = featuresets[100:], featuresets[:100]
classifier = NaiveBayesClassifier.train(train_set)

# Test the classifier
accuracy = nltk.classify.accuracy(classifier, test_set)
print("Accuracy:", accuracy)
