import torch.utils.data
import pandas as pd
import pickle
from os.path import join
import torch


class initProcessing(torch.utils.data.Dataset):

    def __init__(self, args, is_train=True):
        super(initProcessing, self).__init__()
        with open(join(args.preprocessed_data_location, "word2idx.pkl"), "rb") as fh:
            self.word2idx = pickle.load(fh)
        self.is_train = is_train
        pd_text = pd.read_pickle(join(args.preprocessed_data_location, "X_train" if is_train else "X_val")).values
        self.y = pd.read_pickle(join(args.preprocessed_data_location, "y_train" if is_train else "y_val")).values
        self.X = torch.zeros((pd_text.shape[0], args.agent_length + args.cust_length), dtype=torch.long)
        for i_sample, sample in enumerate(pd_text):
            for i, sentence in enumerate(sample):  # agent or customer texts.
                for i_token, token in enumerate(sentence.split(" ")):
                    if i_token >= (args.agent_length if i == 0 else args.cust_length):
                        break
                    if token in self.word2idx:
                        self.X[i_sample, i_token + i * args.agent_length] = self.word2idx[token]
                    else:
                        self.X[i_sample, i_token + i * args.agent_length] = self.word2idx["UNK"]

    def get_class_count(self):
        return self.y.shape[1]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx].nonzero()[0].tolist() if self.is_train else self.y[idx].tolist()
