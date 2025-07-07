import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Sample text
text = "ChatGPT is a language model developed by OpenAI."

# Tokenize the text into words
words = word_tokenize(text)

# Perform part-of-speech tagging
pos_tags = pos_tag(words)

print(pos_tags)

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

nltk.download('punkt')

# Tokenizing into words
text = "Tokenization is important for NLP tasks."
words = word_tokenize(text)
print(words)

# Tokenizing into sentences
text = "Tokenization is important. It helps break text into sentences."
sentences = sent_tokenize(text)
print(sentences)

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

text = "Corpus preprocessing involves removing punctuation, stop words, and lemmatizing words."

# Tokenization
words = word_tokenize(text)

# Lowercasing
words = [word.lower() for word in words]

# Removing punctuation
words = [word for word in words if word.isalnum()]

# Stopword removal
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words if word not in stop_words]

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]

print(lemmatized_words)
