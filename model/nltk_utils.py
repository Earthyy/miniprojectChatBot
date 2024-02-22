import numpy as np
import nltk
#nltk.download('punkt')
from pythainlp import word_tokenize

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    # sentence =' '.join(sentence) #ถ้าทำการเเยกคำต้องนี้จะเป็นการเเยกตัวอักษรเลย เเต่ก็ได้ผลที่ดีมาก
    return word_tokenize(sentence)


def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
    # stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    # print("\n", sentence_words)
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag