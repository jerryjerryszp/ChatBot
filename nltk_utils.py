import nltk
import numpy as np

from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()
# nltk.download('punkt')


# tokenize the sentence
def tokenize(sentence):
    return nltk.word_tokenize(sentence)


# stem the word
def stem(word):
    return stemmer.stem(word.lower())


# create the bag of words
def bagOfWords(tokenizedSentence, allWords):
    tokenizedSentence = [stem(word) for word in tokenizedSentence]
    bag = np.zeros(len(allWords), dtype=np.float32)

    for index, word in enumerate(allWords):
        if word in tokenizedSentence:
            bag[index] = 1.0

    return bag
