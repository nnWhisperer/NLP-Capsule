#!/usr/bin/env python
# coding: utf-8
import pandas as pd
from sklearn.model_selection import train_test_split
# from collections import Counter
# import ujson as json
import sys
from args import get_preprocessing_args
import os


def get_file_contents(test_or_train, agent_or_cust_or_target, dir_location='./'):
    return pd.read_csv(os.path.join(dir_location, "Koc_Yaz_Okulu_Data_" + test_or_train + agent_or_cust_or_target + ".txt"), sep=";", header=0)


def main(main_args):
    args = get_preprocessing_args(main_args)
    print(args)

    str_test = "Test_"
    str_train = "Train_"
    str_agent = "Agent"
    str_cust = "Cust"
    str_target = "Target"

    if not os.path.isdir(args.input_data_location):
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

    X_train, X_val, y_train, y_val = train_test_split(merged_train.iloc[:, :2], merged_train.iloc[:, 2:], test_size=0.2, random_state=1)
    X_train.reset_index(inplace=True)
    X_train.drop("index", axis=1, inplace=True)
    X_val.reset_index(inplace=True)
    X_val.drop("index", axis=1, inplace=True)
    y_train.reset_index(inplace=True)
    y_train.drop("index", axis=1, inplace=True)
    y_val.reset_index(inplace=True)
    y_val.drop("index", axis=1, inplace=True)

    X_train.to_pickle(args.preprocessed_data_location + "X_train")
    temp = pd.read_pickle(args.preprocessed_data_location + "X_train")
    assert X_train.equals(temp)

    X_val.to_pickle(args.preprocessed_data_location + "X_val")
    temp = pd.read_pickle(args.preprocessed_data_location + "X_val")
    assert X_val.equals(temp)

    y_train.to_pickle(args.preprocessed_data_location + "y_train")
    temp = pd.read_pickle(args.preprocessed_data_location + "y_train")
    assert y_train.equals(temp)

    y_val.to_pickle(args.preprocessed_data_location + "y_val")
    temp = pd.read_pickle(args.preprocessed_data_location + "y_val")
    assert y_val.equals(temp)

    X_test = test_agent.join(test_cust)
    X_test.reset_index(inplace=True)
    X_test.drop("ID", axis=1, inplace=True)

    X_test.to_pickle(args.preprocessed_data_location + "X_test")
    temp = pd.read_pickle(args.preprocessed_data_location + "X_test")
    assert X_test.equals(temp)

    # agent_vocabulary = Counter()
    # cust_vocabulary = Counter()

    # for i in merged_train["AGENT_TXT"]:
    #     agent_vocabulary.update(i.split(' '))

    # for i in merged_train["CUST_TXT"]:
    #     cust_vocabulary.update(i.split(' '))

    # word2idx = {}
    # for i, key in enumerate(set(cust_vocabulary.keys()).union(set(agent_vocabulary.keys()))):
    #     word2idx[key] = i

    # assert set(agent_vocabulary.keys()) - set(word2idx.keys()) == set()
    # assert set(cust_vocabulary.keys()) - set(word2idx.keys()) == set()

    # with open(args.preprocessed_data_location + "word2idx.json", "w") as fh:
    #     json.dump(word2idx, fh)

    # with open(args.preprocessed_data_location + "word2idx.json", "r") as fh:
    #     temp = json.load(fh)

    # assert temp == word2idx


if __name__ == '__main__':
    main(sys.argv[1:])
