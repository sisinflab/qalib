import nltk

from os import makedirs
from os.path import exists, join


def create(dir):
    if not exists(dir):
        makedirs(dir)


def add_newFAQs(id, filename):
    create(join("qalib/utils", id))
    create(join("qalib/models", id, "models_word2tag"))
    with open(filename) as src:
        lines = src.read()
    with open(join("qalib/utils", id, "nice_faq"), 'w') as dst:
        dst.write(lines)


def init():

    create("qalib/models")
    create("qalib/utils")

    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw')
