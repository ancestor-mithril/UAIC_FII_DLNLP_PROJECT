import os
import re
import string

import nltk.corpus
import numpy as np
import pandas as pd
from collections import Counter

from nltk import word_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm


def make_dataframe(input_folder, labels_fn=None):
    # MAKE TXT DATAFRAME
    text = []
    for fil in tqdm(filter(lambda x: x.endswith('.txt'), os.listdir(input_folder))):
        iD = fil[7:].split('.')[0]
        lines = list(enumerate(open(input_folder + fil, 'r', encoding='utf-8').read().splitlines(), 1))
        text.extend([(iD,) + line for line in lines])

    df_text = pd.DataFrame(text, columns=['id', 'line', 'text'])
    df_text.id = df_text.id.apply(int)
    df_text.line = df_text.line.apply(int)
    df_text = df_text[df_text.text.str.strip().str.len() > 0].copy()
    df_text = df_text.set_index(['id', 'line'])

    df = df_text

    if labels_fn:
        # MAKE LABEL DATAFRAME
        labels = pd.read_csv(labels_fn, sep='\t', encoding='utf-8', header=None)
        labels = labels.rename(columns={0: 'id', 1: 'line', 2: 'labels'})
        labels = labels.set_index(['id', 'line'])
        labels = labels[labels.labels.notna()].copy()

        # JOIN
        df = labels.join(df_text)[['text', 'labels']]

    return df


def get_labels():
    return [line.rstrip() for line in open("./data/scorers/techniques_subtask3.txt", encoding='utf-8').readlines()]


def get_all_languages():
    return [x for x in os.listdir('./data/data')]


def load_train_and_dev(language):
    test_dataset = make_dataframe(f"./data/data/{language}/train-articles-subtask-3/",
                                  f"./data/data/{language}/train-labels-subtask-3.txt")
    dev_dataset = make_dataframe(f"./data/data/{language}/dev-articles-subtask-3/")
    return test_dataset, dev_dataset


def get_language(language):
    available = {
        'ca': 'catalan',
        'cs': 'czech',
        'de': 'german',
        'el': 'greek',
        'en': 'english',
        'es': 'spanish',
        'fi': 'finnish',
        'fr': 'french',
        'hu': 'hungarian',
        'is': 'icelandic',
        'it': 'italian',
        'lv': 'latvian',
        'nl': 'dutch',
        'pl': 'polish',
        'pt': 'portuguese',
        'ro': 'romanian',
        'ru': 'russian',
        'sk': 'slovak',
        'sl': 'slovenian',
        'sv': 'swedish',
        'ta': 'tamil'
    }
    return available.get(language, language)  # default is the same


def build_stop_words(language):
    stop_words = set(stopwords.words(get_language(language)))
    punctuation = string.punctuation + '-' + '+' + '—' + '„' + "”" + '–' + '“' + '’' + '‘'
    special_tokens = ["'s", "'m", "", "``", '\'\'', ]
    return list(stop_words) + list(punctuation) + special_tokens


def tokenize(dataset, remove):
    words = Counter()
    for i, row in dataset.iterrows():
        tokens = []
        for word in word_tokenize(row["text"]):
            word = word.lower()
            if word in remove:
                continue
            words.update([word])
            tokens.append(word)
        row["text"] = tokens

    return words


def get_word2idx(words):
    # Removing the words that only appear once
    words = {k: v for k, v in words.items() if v > 1}
    # Sorting the words according to the number of appearances, with the most common word being first
    words = sorted(words, key=words.get, reverse=True)
    # Adding padding and unknown to our vocabulary so that they will be assigned an index
    words = ['_PAD', '_UNK'] + words
    # Dictionary to store the word to index mappings
    word2idx = {o: i for i, o in enumerate(words)}
    return word2idx


def sentences2indices(dataset, word2idx):
    for i, row in dataset.iterrows():
        row["text"] = [word2idx.get(word, word2idx['_UNK']) for word in row["text"]]


def pad_input(dataset, seq_len):
    """
    Shortens the sentences or pads them with 0.
    :param dataset: Dataframe with sentences in "text"
    :param seq_len: The length of the sequence
    :return:
    """
    for index, row in enumerate(dataset.iterrows()):
        features = np.zeros(seq_len, dtype=int)
        key, row = row
        if len(row["text"]) != 0:
            # TODO: Padding at front? Why not padding at back
            features[-len(row["text"]):] = np.array(row["text"])[:seq_len]
        row["text"] = features


def labels_to_multi_hot(dataset, labels):
    for i, row in dataset.iterrows():
        multihot = np.zeros(len(labels))
        for l in row["labels"].split(","):
            multihot[labels.index(l)] = 1.0
        row["labels"] = multihot


def preprocess(dataset, language):
    remove = build_stop_words(language)

    words = tokenize(dataset, remove)

    word2idx = get_word2idx(words)

    sentences2indices(dataset, word2idx)

    pad_input(dataset, 90)  # TODO: make this a parameter
    # TODO: get a good padding size

    labels_to_multi_hot(dataset, get_labels())

    return word2idx


def get_preprocessed(language):
    df, _ = load_train_and_dev("en")
    word2idx = preprocess(df, language)
    return df, word2idx


def main():
    df1, df2 = load_train_and_dev("en")
    print(df1)
    print(df2)
    print(get_all_languages())
    preprocess(df1, "en")
    print(df1)


if __name__ == "__main__":
    main()
