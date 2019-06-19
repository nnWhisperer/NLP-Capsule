import torch.utils.data
from pickle import load
from os.path import join
import torch


class initProcessing(torch.utils.data.Dataset):

    def __init__(self, args, is_train=True):
        super(initProcessing, self).__init__()
        self.is_train = is_train

        def tempLoad(location, strname):
            with open(join(location, strname + ".pkl"), "rb") as fh:
                temp = load(fh)
            return temp
        self.X = tempLoad(args.preprocessed_data_location, "X_train" if is_train else "X_val")
        self.y = tempLoad(args.preprocessed_data_location, "y_train" if is_train else "y_val")

    def get_class_count(self):
        return self.y.shape[1]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
