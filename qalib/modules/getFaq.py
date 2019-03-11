import numpy as np
from . import preprocessing as p
from . import ann as model
from os.path import exists
from sklearn.metrics.pairwise import cosine_similarity as cs


# FUNCTIONS


def get_FAQ(filename, origin_filename):

    if not exists(filename):

        questions = p.load_preprocessed_questions(origin_filename)

        questions = p.get_tokens(questions)

        original_tokens = []
        for q in questions:
            original_tokens += q

        words = list(set([token.lemma for question in questions for token in question]))

        np.save(filename, [questions, words, original_tokens])

        return questions, words, original_tokens

    else:

        questions, words, original_tokens = np.load(filename).tolist()

        return questions, words, original_tokens


def get_syn(path, words, language):

    syns = []
    if exists(path):
        syns = p.load_syn(path)
    else:
        syns = p.get_syns(words, language)
        p.save_syn(path, syns)

    return syns


def get_vocab(path, syn):

    vocab = []
    if exists(path):
        vocab = np.load(path)
    else:
        vocab = p.get_vocab(syn)
        np.save(path, vocab)

    return vocab


def get_originals(path, syn):

    orginals = []
    if exists(path):
        originals = np.load(path).tolist()
    else:
        originals = list(syn.keys())
        np.save(path, originals)
    return originals


def train(id_device, words, syn, vocab, word_index):

    input = np.zeros((len(vocab), len(vocab)))
    np.fill_diagonal(input, 1)

    output = np.zeros((len(vocab), len(words)))

    for j, word in enumerate(words):
        i = word_index[word]
        output[i, j] = 1
        for synonym in syn[word]:
            if len(synonym) > 0:
                i = word_index[synonym]
                output[i, j] = 1

    model.train(input, output, "qalib/models/{}/models_word2tag".format(id_device))


def extract_words(id_device, questions, filename, vocab, word_index, originals, originals_reverse_index, save=False):

    save_questions = []
    for question in questions:

        question = p.preprocess_question(question)
        tokens = p.get_token(question)

        save_question = []
        for token in tokens:
            if token.lemma in vocab:
                array = np.zeros((1, len(vocab)))
                array[0, word_index[token.lemma]] = 1

                values = model.predict(array, len(originals), "qalib/models/{}/models_word2tag".format(id_device))

                position = np.argmax(values)

                if values[position] > 0.8:
                    save_question.append(originals_reverse_index[position])

                # print(token.lemma, originals_reverse_index[position], values[position])

        save_questions.append(save_question)

    if save:
        with open(filename, 'w') as file:
            file.write('\n'.join('::'.join(q) for q in save_questions))

        # print('\n'.join(' '.join(q) for q in save_questions))

    return save_questions


def get_extracted(exs, original_tokens):
    extracted = []
    for ex in exs:
        sub_exs = []
        for sub_ex in ex:
            for t in original_tokens:
                if t.lemma == sub_ex:
                    sub_exs.append(t)
                    break
        extracted.append(sub_exs)
    return extracted


def get_question_matrix(filename, questions, tags):

    index = dict((tag, count) for count, tag in enumerate(tags))

    questions_matrix = []
    if exists(filename):
        questions_matrix = np.load(filename)
    else:
        questions_matrix = np.zeros((len(questions), len(tags)))
        for i, question in enumerate(questions):
            for token in question:
                j = index[token.lemma]
                questions_matrix[i, j] = token.depth
        np.save(filename, questions_matrix)

    return index, questions_matrix


def cos_sim(filename, extracted, questions, tags):

    index, questions_matrix = get_question_matrix(filename, questions, tags)

    token_matrix = np.zeros((1, len(tags)))
    for token in extracted:
        j = index[token.lemma]
        token_matrix[0, j] = token.depth

    values = cs(token_matrix, questions_matrix)[0]

    position = np.argmax(values)

    # print(position, values[position], [t.text for t in question], [t.text for t in questions[position]])

    return values[position], position + 1

########################################################################################################################


def findFAQ(id_device, user_question=None, language="it", training=False):

    if not training:
        assert user_question, "No questions provided."

    p.init(language)

    questions, words, original_tokens = get_FAQ("qalib/utils/{}/data_faq.npy".format(id_device), "qalib/utils/{}/nice_faq".format(id_device))

    syn = get_syn("qalib/utils/{}/syns".format(id_device), words, language)

    vocab = get_vocab("qalib/utils/{}/vocab.npy".format(id_device), syn)

    word_index = dict((w, i) for i, w in enumerate(vocab))

    originals = get_originals("qalib/utils/{}/originals.npy".format(id_device), syn)

    if training:
        train(id_device, originals, syn, vocab, word_index)
        # exit()

    originals_index = dict((w, c) for c, w in enumerate(originals))
    originals_reverse_index = dict((c, w) for c, w in enumerate(originals))

    if training:
        questions = p.load_preprocessed_questions("qalib/utils/{}/nice_faq".format(id_device))
        extract_words(id_device, questions, "qalib/utils/{}/faq_tag".format(id_device), vocab, word_index, originals, originals_reverse_index, save=True)
        # exit()

    exs_originals = p.load_extracted("qalib/utils/{}/faq_tag".format(id_device))
    extracted_originals = get_extracted(exs_originals, original_tokens)

    if training:
        get_question_matrix("qalib/utils/{}/question_matrix.npy".format(id_device), extracted_originals, originals)
        return

    new_questions = [user_question]
    exs = extract_words(id_device, new_questions, "", vocab, word_index, originals, originals_reverse_index, save=False)
    extracted = get_extracted(exs, original_tokens)[0]

    value, index = cos_sim("qalib/utils/{}/question_matrix.npy".format(id_device), extracted, extracted_originals, originals)

    result = '{{"result": "{}", "index": "{}"}}'
    # result = '{{"result": "{}", "index": "{}", "question:", "{}", "answer:", " {}"}}'

    if value > 0.8:
        result = result.format("success", index)
    else:
        if value > 0.5:
            result = result.format("mean", index)
        else:
            result = result.format("error", index)

    print(result, end="")
    
    return result

# findFAQ(id_device, user_question, training=False)
