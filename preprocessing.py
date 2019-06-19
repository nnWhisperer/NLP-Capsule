#!/usr/bin/env python
# coding: utf-8
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
from args import get_preprocessing_args
import os
from os.path import join, isdir
from making_our_word2vec import construct_word2vec, save_and_assert, save_and_assert_2d
import numpy as np


def lookUpColumn(trim_length, aColumn, word2idx):
    temp = aColumn.str.split(" ").apply(lambda a: list(map(lambda key: word2idx[key] if key in word2idx else word2idx["UNK"], a)))
    newArray = np.zeros((len(temp), trim_length), 'long')
    for i, j in enumerate(temp):
        max_indice = min(len(j), trim_length)
        newArray[i][0:max_indice] = j[:max_indice]
    return newArray


def lookUpColumns(a_pd_frame, word2idx, agent_length, cust_length):
    b = lookUpColumn(agent_length, a_pd_frame["AGENT_TXT"], word2idx)
    c = lookUpColumn(cust_length, a_pd_frame["CUST_TXT"], word2idx)
    assert c.shape[0] == b.shape[0]
    return np.hstack((b, c))


def get_file_contents(test_or_train, agent_or_cust_or_target, dir_location='./'):
    return pd.read_csv(join(dir_location, "Koc_Yaz_Okulu_Data_" + test_or_train + agent_or_cust_or_target + ".txt"), sep=";", header=0)


def main(main_args):
    args = get_preprocessing_args(main_args)
    print(args)

    str_test = "Test_"
    str_train = "Train_"
    str_agent = "Agent"
    str_cust = "Cust"
    str_target = "Target"

    if not isdir(args.input_data_location):
        raise Exception(args.input_data_location + " is not a directory.")
    os.makedirs(args.preprocessed_data_location, exist_ok=True)

    test_agent = get_file_contents(str_test, str_agent, dir_location=args.input_data_location)
    test_cust = get_file_contents(str_test, str_cust, dir_location=args.input_data_location)
    train_agent = get_file_contents(str_train, str_agent, dir_location=args.input_data_location)
    train_cust = get_file_contents(str_train, str_cust, dir_location=args.input_data_location)
    train_target = get_file_contents(str_train, str_target, dir_location=args.input_data_location)
    train_target.set_index("ID", inplace=True)
    train_agent.set_index("ID", inplace=True)
    train_cust.set_index("ID", inplace=True)
    test_agent.set_index("ID", inplace=True)
    test_cust.set_index("ID", inplace=True)

    merged_train = train_agent.join(train_cust)
    merged_train = merged_train.join(train_target)

    merged_train.reset_index(inplace=True)
    merged_train.drop("ID", axis=1, inplace=True)

    values = {}
    if not os.path.exists(join(args.preprocessed_data_location, "word2idx.pkl")):
        values["word2idx"], values["embedding_weights"] = construct_word2vec(args, merged_train)

        save_and_assert(values, "word2idx", args.preprocessed_data_location)
        save_and_assert_2d(values, "embedding_weights", args.preprocessed_data_location)
    else:
        from pickle import load
        with open(join(args.preprocessed_data_location, "word2idx.pkl"), "rb") as fh:
            values["word2idx"] = load(fh)
    X = lookUpColumns(merged_train, values["word2idx"], args.agent_length, args.cust_length)

    values["X_train"], values["X_val"], values["y_train"], values["y_val"] = train_test_split(X, merged_train.iloc[:, 2:].values, test_size=0.2, random_state=1)

    def save_and_check(name):
        save_and_assert_2d(values, name, args.preprocessed_data_location)

    save_and_check("X_train")
    save_and_check("X_val")
    save_and_check("y_train")
    save_and_check("y_val")

    X_test = test_agent.join(test_cust)
    X_test.reset_index(inplace=True)
    X_test.drop("ID", axis=1, inplace=True)

    values["X_test"] = lookUpColumns(X_test, values["word2idx"], args.agent_length, args.cust_length)
    save_and_check("X_test")


if __name__ == '__main__':
    main(sys.argv[1:])
