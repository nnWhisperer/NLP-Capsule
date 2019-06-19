#!/usr/bin/env python
# coding: utf-8
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
import pickle
from args import get_preprocessing_args
import sys
from os.path import join


class sentence_generator():
    def __init__(self, dataset):
        self.n = len(dataset)
        self.dataset = dataset
        print("started")

    def __iter__(self):
        return self

    def __next__(self):
        if self.n > 0:
            self.n -= 1
            return self.dataset.loc[self.n][0].split(" ")
        elif self.n == 0:
            self.n -= 1
            return ["UNK"]
        else:
            print("reached the end of the sentence_generator.")
            self.n = len(self.dataset)
            raise StopIteration


def construct_word2vec(args, corpus):
    merged_agent_cust = corpus["CUST_TXT"] + " " + corpus["AGENT_TXT"] + " " + corpus["CUST_TXT"] + " " + corpus["AGENT_TXT"]
    merged_sentences = sentence_generator(merged_agent_cust.reset_index().drop("index", axis=1))
    model = Word2Vec(min_count=1, size=args.vec_size, window=5)
    model.build_vocab(merged_sentences)  # prepare the model vocabulary
    model.train(merged_sentences, total_examples=model.corpus_count, epochs=5)

    embedding_weights = np.zeros((len(model.wv.vocab.keys()), args.vec_size), 'float32')

    word2idx = {}
    for i, key in enumerate(model.wv.vocab.keys()):
        word2idx[key] = i
        embedding_weights[i] = np.array(model.wv[key].copy())

    try:
        temp = word2idx["UNK"]
        assert type(temp) == int
    except KeyError:
        raise Exception("there was an error constructing the UNK token in Word2Vec.")

    assert embedding_weights.shape == (len(model.wv.vocab.keys()), args.vec_size)
    assert all((embedding_weights[word2idx['X7092177']] != np.zeros((1, args.vec_size)))[0])
    return word2idx, embedding_weights


def save_and_load(values, strname, preprocessed_data_location):
    with open(join(preprocessed_data_location, strname + ".pkl"), "wb") as fh:
        pickle.dump(values[strname], fh)

    with open(join(preprocessed_data_location, strname + ".pkl"), "rb") as fh:
        temp = pickle.load(fh)
    return temp


def save_and_assert(values, strname, preprocessed_data_location):
    temp = save_and_load(values, strname, preprocessed_data_location)
    assert all([temp[i] == values[strname][i] for i in values[strname].keys()])


def save_and_assert_2d(values, strname, preprocessed_data_location):
    temp = save_and_load(values, strname, preprocessed_data_location)
    assert all(all(temp[i] == values[strname][i]) for i in range(values[strname].shape[0]))


def main(main_args):
    args = get_preprocessing_args(main_args)
    print(args)

    X_train = pd.read_pickle(join(args.preprocessed_data_location, "X_train"))
    values = {}
    values["word2idx"], values["embedding_weights"] = construct_word2vec(args, X_train)

    save_and_assert(values, "word2idx", args.preprocessed_data_location)
    save_and_assert_2d(values, "embedding_weights", args.preprocessed_data_location)


if __name__ == '__main__':
    main(sys.argv[1:])
