import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Sample text
text = "Parts of speech tagging helps analyze text."

# Tokenize the text
words = word_tokenize(text)

# Perform parts of speech tagging
tags = pos_tag(words)

print(tags)


# Load the English NLP model
nlp = spacy.load("en_core_web_sm")

# Sample text
text = "Parts of speech tagging helps analyze text."

# Process the text with spaCy
doc = nlp(text)

# Iterate through tokens and access their part of speech tags
for token in doc:
    print(token.text, token.pos_)
