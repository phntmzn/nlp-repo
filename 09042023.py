import nltk

def parse_gender(text):
    # Tokenize the text into sentences and words
    sentences = [
        [word.lower() for word in nltk.word_tokenize(sentence)] 
        for sentence in nltk.sent_tokenize(text)
    ]

    # Call the count_gender function to count gender-related words
    sents, words = count_gender(sentences)
    
    # Calculate the total count of gender-related words
    total = sum(words.values())
    
    # Print the results
    for gender, count in words.items():
        pcent = (count / total) * 100
        nsents = sents[gender]
        print("{:.3f}% {} ({} sentences)".format(pcent, gender, nsents))

# You would need to define the count_gender function separately
# since it's not provided in the code you shared.
