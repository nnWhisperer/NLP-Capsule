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


def main(main_args):
    args = get_preprocessing_args(main_args)
    print(args)

    X_train = pd.read_pickle(join(args.preprocessed_data_location, "X_train"))
    merged_agent_cust = X_train["CUST_TXT"] + " " + X_train["AGENT_TXT"] + " " + X_train["CUST_TXT"] + " " + X_train["AGENT_TXT"]

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

    with open(join(args.preprocessed_data_location, "word2idx.pkl"), "wb") as fh:
        pickle.dump(word2idx, fh)

    temp = None
    with open(join(args.preprocessed_data_location, "word2idx.pkl"), "rb") as fh:
        temp = pickle.load(fh)

    assert all([temp[i] == word2idx[i] for i in word2idx.keys()])

    with open(join(args.preprocessed_data_location, "embedding_weights.pkl"), "wb") as fh:
        pickle.dump(embedding_weights, fh)

    with open(join(args.preprocessed_data_location, "embedding_weights.pkl"), "rb") as fh:
        temp = pickle.load(fh)

    assert all(all(temp[i] == embedding_weights[i]) for i in range(embedding_weights.shape[0]))


if __name__ == '__main__':
    main(sys.argv[1:])
