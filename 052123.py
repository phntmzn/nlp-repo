import re
from nltk.tokenize import word_tokenize

# 1. Using the split() method
text = "Hello, how are you?"
tokens = text.split()  # Splitting by whitespace
print(tokens)
# Output: ['Hello,', 'how', 'are', 'you?']

# 2. Using regular expressions (re module)
text = "Hello, how are you?"
tokens = re.findall(r'\w+', text)  # Splitting based on one or more word characters
print(tokens)
# Output: ['Hello', 'how', 'are', 'you']

# 3. Using Natural Language Processing libraries
from nltk.tokenize import word_tokenize

text = "Hello, how are you?"
tokens = word_tokenize(text)
print(tokens)


import spacy

nlp = spacy.load('en_core_web_sm')
text = "Hello, how are you?"
doc = nlp(text)
tokens = [token.text for token in doc]
print(tokens)
