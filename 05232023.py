import re
import nltk
from nltk.tokenize import word_tokenize

# Preprocess and tokenize the input phrases
def preprocess_phrases(phrase):
    # Normalize the text by converting to lowercase
    normalized_phrase = phrase.lower()

    # Tokenize the normalized phrase into words
    tokens = word_tokenize(normalized_phrase)
    return tokens

# Load lexicon or dictionary of words
def load_lexicon():
    lexicon = set()

    # Add words from a text file to the lexicon
    with open('lexicon.txt', 'r') as file:
        for line in file:
            word = line.strip()
            lexicon.add(word)

    return lexicon

# Generate a response based on the input phrases
def generate_response(input_phrases):
    lexicon = load_lexicon()
    response = ""

    # Process each input phrase
    for phrase in input_phrases:
        tokens = preprocess_phrases(phrase)

        # Find relevant words using regex pattern matching
        relevant_words = []
        for token in tokens:
            if re.match(r'^[a-zA-Z]+$', token):  # Only consider alphabetic words
                if token in lexicon:
                    relevant_words.append(token)

        # Construct the response by combining the relevant words
        response += ' '.join(relevant_words) + ' '

    # Truncate the response to 120 characters if needed
    if len(response) > 120:
        response = response[:120]

    return response

# Example usage
input_phrases = ["I enjoy coding in Python.", "Machine learning is fascinating."]
generated_response = generate_response(input_phrases)
print(generated_response)
