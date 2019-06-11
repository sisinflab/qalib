import spacy
import re

import numpy as np
from . import retrieve_synonyms_script as synonym

from os.path import exists

from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

# initialization


def init(lang):

    global nlp
    global stemmer
    global lemmatizer
    global stopWords

    zipcode = "it"
    language = "italian"

    if lang == "en":
        zipcode = "en"
        language = "english"

    nlp = spacy.load(zipcode)

    stemmer = SnowballStemmer(language)
    lemmatizer = WordNetLemmatizer()
    stopWords = set(stopwords.words(language))

# functions


class Token:

    text = ""
    lemma = ""
    depth = ""

    def __init__(self, text, lemma, depth):
        self.text = text
        self.lemma = lemma
        self.depth = depth

    def __str__(self):
        return "WORD:" + self.text + "LEMMA:" + self.lemma + "DEPTH:" + self.depth


def load_questions(filename):
    with open(filename) as file:
        lines = [line for line in file.read().splitlines() if "?" in line]
    return lines


def preprocess_question(question):
    question = re.sub(r'[’|\']', ' ', question.lower())
    question = re.sub(r'[^\w\s]', '', question)
    return question


def load_preprocessed_questions(filename):
    lines = load_questions(filename)
    questions = [preprocess_question(question) for question in lines]
    return questions


def lemmatization(question):
    question = re.sub(r'[’|\']', ' ', question.lower())
    question = re.sub(r'[^\w\s]', '', question)
    words = set()
    sentence = nlp(question)
    for token in sentence:
        if not token.is_stop:
            words.add(token.lemma_)
    return list(words)


def ordered_lemmatization(question):
    question = re.sub(r'[’|\']', ' ', question.lower())
    question = re.sub(r'[^\w\s]', '', question)
    words = []
    sentence = nlp(question)
    for token in sentence:
        if not token.is_stop:
            words.append(token.lemma_)
    return ' '.join(words)


def load_sentences(filename):
    with open(filename) as file:
        lines = file.read().splitlines()
    questions = [re.sub(r'[’|\']', ' ', line.lower()) for line in lines if "?" in line]
    questions = [re.sub(r'[^\w\s]', '', question) for question in questions]
    return questions


def get_words(questions):
    words = set()
    for question in questions:
        sentence = nlp(question)
        for token in sentence:
            if not token.is_stop:
                words.add(token.lemma_)
    return list(words)


def entity(filename):
    with open(filename) as file:
        lines = file.read().splitlines()
    return dict((line.split("\t")[0], line.split("\t")[1]) for line in lines if "NR" not in line.split("\t")[1])


def get_syns(words, language):
    syn = {}
    for k in words:
        syn[k] = synonym.retrieve_syn_list(k, language)
    return syn


def get_vocab(syn):
    vocab = set()
    vocab.update(list(syn.keys()))
    for l in syn.values():
        vocab.update(l)
    vocab = list(vocab)

    # np.save("utils/vocab", vocab)
    return vocab


def create_vocab(d, syn):
    vocab = set()
    vocab.update(list(d.keys()))
    for l in syn.values():
        vocab.update(l)
    vocab = list(vocab)

    np.save("qalib/utils/vocab", vocab)


def load_syn(filename):
    with open(filename) as file:
        lines = file.read().splitlines()
    syn = {}
    for line in lines:
        k, syns = line.split("\t")
        syn[k] = syns.split(",")
    return syn


def save_syn(filename, syn):
    lines = []
    for k in syn:
        line = "{}\t".format(k)
        for s in syn[k]:
            line += "{},".format(s)
        if len(syn[k]) > 0:
            line = line[:-1]
        lines.append(line)

    with open(filename, 'w') as file:
        file.write("\n".join(lines))


def write_syn_file(d, syn):
    lines = []
    for k in d:
        line = "{}\t".format(d[k])
        for s in syn[k]:
            line += "{},".format(s)
        if len(syn[k]) > 0:
            line = line[:-1]
        lines.append(line)

    with open("qalib/utils/syns", 'w') as file:
        file.write("\n".join(lines))


def load_vocab():
    return np.load("qalib/utils/vocab.npy").tolist()


def load_tags():
    return np.load("qalib/utils/tags.npy").tolist()


def load_data(filename):
    with open(filename) as file:
        lines = file.read().splitlines()

    tag_terms = {}
    for line in lines:
        tag, terms = line.split("\t")
        if tag not in tag_terms:
            tag_terms[tag] = []
        tag_terms[tag] += terms.split(",")

    terms_tag = {}
    for k in tag_terms:
        for w in tag_terms[k]:
            if w not in terms_tag:
                terms_tag[w] = set()
            terms_tag[w].add(k)

    with open("qalib/utils/tags.tsv") as file:
        lines = file.read().splitlines()
    lines = [line for line in lines if "NR" not in line]
    for line in lines:
        term, tag = line.split("\t")
        if term not in terms_tag:
            terms_tag[term] = set()
        terms_tag[term].add(tag)

    terms_tag = dict((k, list(terms_tag[k])) for k in terms_tag)

    tags = list(tag_terms.keys())
    if not exists("qalib/utils/tags.npy"):
        np.save("qalib/utils/tags", tags)
    else:
        tags = np.load("qalib/utils/tags.npy").tolist()

    return tags, load_vocab(), terms_tag


def get_token(sentence):

    def get(node, count, tokens):

        if node.text not in stopWords:

            lemmas = wn.lemmas(node.text, lang="ita")
            lemma = lemmas[0].name() if len(lemmas) > 0 else node.lemma_

            token = Token(node.text, lemma, count)

            tokens.append(token)

        if node.n_lefts + node.n_rights > 0:
            count += 1
            [get(child, count, tokens) for child in node.children]

        return tokens

    doc = nlp(sentence)
    tokens = []
    [get(sent.root, 1, tokens) for sent in doc.sents]

    return tokens


def get_tokens(questions):

    questions_tokens = []

    for sentence in questions:

        questions_tokens.append(get_token(sentence))

    return questions_tokens


def load_extracted(filename):
    with open(filename) as file:
        lines = file.read().splitlines()
    extracted = []
    for line in lines:
        extracted.append((token.split(':')[0], int(token.split(':')[1])) for token in line.split("::"))
    return extracted
