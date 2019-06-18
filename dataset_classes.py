import torch.utils.data
import data_helpers


class simpleDataset(torch.utils.data.Dataset):

    def __init__(self, args):
        super(simpleDataset, self).__init__()
        self.X_trn, self.Y_trn, self.Y_trn_o, self.X_tst, self.Y_tst, self.Y_tst_o, self.vocabulary, self.vocabulary_inv = data_helpers.load_data(args.dataset, max_length=args.sequence_length, vocab_size=args.vocab_size)
        self.is_train = True

    def get_class_count(self):
        return self.Y_trn.shape[1]

    def get_vocabulary_inv(self):
        return self.vocabulary_inv

    def __len__(self):
        return self.X_trn.shape[0] if self.is_train else self.X_tst.shape[0]

    def set_train(self):
        self.is_train = True

    def set_val(self):
        self.is_train = False

    def __getitem__(self, idx):
        return (self.X_trn[idx], self.Y_trn_o[idx]) if self.is_train else (self.X_tst[idx], self.Y_tst[idx])
