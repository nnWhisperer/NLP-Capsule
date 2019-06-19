import torch.utils.data
import pandas as pd
import pickle
from os.path import join
import torch


class dynamicProcessing(torch.utils.data.Dataset):

    def __init__(self, args):
        super(dynamicProcessing, self).__init__()
        with open(join(args.preprocessed_data_location, "word2idx.pkl"), "rb") as fh:
            self.word2idx = pickle.load(fh)
        self.X_train = pd.read_pickle(join(args.preprocessed_data_location, "X_train")).values
        self.X_val = pd.read_pickle(join(args.preprocessed_data_location, "X_val")).values
        self.y_train = pd.read_pickle(join(args.preprocessed_data_location, "y_train")).values
        self.y_val = pd.read_pickle(join(args.preprocessed_data_location, "y_val")).values
        self.agent_length = args.agent_length
        self.cust_length = args.cust_length
        self.is_train = True

    def get_class_count(self):
        return self.y_train.shape[1]

    def get_vocabulary_inv(self):
        return None

    def __len__(self):
        return self.X_train.shape[0] if self.is_train else self.X_val.shape[0]

    def set_train(self):
        self.is_train = True

    def set_val(self):
        self.is_train = False

    def __getitem__(self, idx):
        return self.process(self.X_train[idx], self.y_train[idx]) if self.is_train else self.process(self.X_val[idx], self.y_val[idx])

    def process(self, X_item, Y_item):
        X = torch.zeros(self.agent_length + self.cust_length, dtype=torch.long)
        for i, sentence in enumerate(X_item):
            for i_token, token in enumerate(sentence.split(" ")):
                if i_token >= (self.agent_length if i == 0 else self.cust_length):
                    break
                if token in self.word2idx:
                    X[i_token + i * self.agent_length] = self.word2idx[token]
                else:
                    X[i_token + i * self.agent_length // 2] = self.word2idx["UNK"]

        return X, Y_item.nonzero()[0].tolist()
