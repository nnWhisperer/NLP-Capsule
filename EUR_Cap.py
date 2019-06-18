from __future__ import division, print_function, unicode_literals
from args import get_cap_args
import numpy as np
import torch
import torch.nn as nn
import os
import json
import time
from torch.optim import Adam
import torch.utils.data
from network import CapsNet_Text, BCE_loss
from w2v import load_word2vec
import data_helpers
import sys
from utils import set_seeds, get_available_devices
from tqdm import tqdm
from tensorboardX import SummaryWriter
import torch.nn.functional as F
"""
TODO: set_lr bir LRscheduler'ina donusturulebilir.
TODO: random seed'i başka bir dosyada set etmek işe yarıyor mu?
TODO: no_grad'lı bir validation bölümü gerekli, validation filan yok.
print'ler ekrana basilmiyorsa, tqdm'le ilgili, squad'da kod var bunun icin.
"""


class simpleDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        super(simpleDataset, self).__init__()
        self.X_trn, self.Y_trn, self.Y_trn_o, self.X_tst, \
            self.Y_tst, self.Y_tst_o, self.vocabulary, self.vocabulary_inv = \
            data_helpers.load_data(args.dataset,
                                   max_length=args.sequence_length,
                                   vocab_size=args.vocab_size)
        self.Y_trn = self.Y_trn.toarray()
        self.Y_tst = self.Y_tst.toarray()

        self.X_trn = self.X_trn.astype(np.int32)
        self.X_tst = self.X_tst.astype(np.int32)
        self.Y_trn = self.Y_trn.astype(np.int32)
        self.Y_tst = self.Y_tst.astype(np.int32)
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


def my_collate_fn(examples):
    return list(zip(*examples))


def is_better_to_stop(epoch, start_time, hours):
    if not hours:  # 0 and 0.0 are to be accepted.
        return False
    approx_hrs_in_seconds = hours * 60 * 60
    seconds_elapsed = time.time() - start_time
    assert seconds_elapsed > 0, "time elapsed can't be negative."
    seconds_per_epoch = seconds_elapsed / epoch
    return (seconds_elapsed + seconds_per_epoch) >= approx_hrs_in_seconds


def main(main_args):
    set_seeds(0)
    a = torch.rand(5)
    print(a)
    # assert all(a == torch.tensor([0.4963, 0.7682, 0.0885, 0.1320, 0.3074]))
    args = get_cap_args(main_args)
    params = vars(args)
    print(json.dumps(params, indent=2))

    validation_step, train_step = 0, 0
    tbx = SummaryWriter(args.save_dir)

    the_dataset = simpleDataset(args)
    dataset_loader = torch.utils.data.DataLoader(the_dataset,
                                                 batch_size=args.tr_batch_size,
                                                 shuffle=True,
                                                 num_workers=args.num_workers,
                                                 collate_fn=my_collate_fn)

    embedding_weights = load_word2vec('glove', the_dataset.get_vocabulary_inv(), args.vec_size)
    args.num_classes = the_dataset.get_class_count()

    device, args.gpu_ids = get_available_devices()
    args.tr_batch_size *= max(1, len(args.gpu_ids))
    model = CapsNet_Text(args, embedding_weights)
    model = nn.DataParallel(model, args.gpu_ids)
    model = model.to(device)
    def transformLabels(labels):
        label_index = list(set([l for _ in labels for l in _]))
        label_index.sort()

        variable_num_classes = len(label_index)
        target = []
        for _ in labels:
            tmp = np.zeros([variable_num_classes], dtype=np.float32)
            tmp[[label_index.index(l) for l in _]] = 1
            target.append(tmp)
        target = torch.tensor(target)
        return label_index, target

    current_lr = args.learning_rate

    optimizer = Adam(model.parameters(), lr=current_lr)

    def set_lr(optimizer, lr):
        for group in optimizer.param_groups:
            group['lr'] = lr

    if args.stop_time:
        start_time = time.time()

    for epoch in range(args.num_epochs):
        if args.stop_time and is_better_to_stop(epoch, start_time, args.stop_time):
            break
        if len(args.gpu_ids) > 0:  # these may not be necessary, let's experiment with it.
            torch.cuda.empty_cache()

        if epoch > args.learning_rate_decay_start and args.learning_rate_decay_start >= 0:
            frac = (epoch - args.learning_rate_decay_start) // args.learning_rate_decay_every
            decay_factor = args.learning_rate_decay_rate ** frac
            current_lr = current_lr * decay_factor
        print(current_lr)
        set_lr(optimizer, current_lr)  # this could be replaced by a scheduler.

        the_dataset.set_train()
        model.train()

        with torch.enable_grad(), tqdm(total=len(dataset_loader.dataset)) as progress_bar:
            for X, Y in dataset_loader:
                start = time.time()
                data = X.long().to(device)

                batch_labels, batch_target = transformLabels(Y)
                batch_target = batch_target.float().to(device)
                optimizer.zero_grad()
                poses, activations = model(data, batch_labels)
                loss = BCE_loss(activations, batch_target)
                loss.backward()
                optimizer.step()
                if len(args.gpu_ids) > 0:
                    torch.cuda.empty_cache()

                progress_bar.update(args.tr_batch_size)
                progress_bar.set_postfix(epoch=epoch, NLL=loss.item(), elapsed=time.time() - start)
                train_step += args.tr_batch_size
                tbx.add_scalar('train/NLL', loss.item(), train_step)

        the_dataset.set_val()
        model.eval()
        losses = []

        with torch.no_grad(), tqdm(total=len(dataset_loader.dataset)) as progress_bar:
            for X, Y in dataset_loader:
                start = time.time()

                data = torch.stack(X, dim=0).to(device)
                poses, activations = model(data, None)
                Y = torch.stack(list(map(lambda a: torch.tensor(a.todense(), dtype=torch.float), Y))).squeeze(1)
                Y = F.pad(Y, (0, activations.shape[1] - Y.shape[1]), "constant", value=0).to(device)
                loss = BCE_loss(activations, Y)
                losses.append(loss)

                progress_bar.update(args.tr_batch_size)
                progress_bar.set_postfix(epoch=epoch, NLL=loss.item(), elapsed=time.time() - start)
                validation_step += args.tr_batch_size
                tbx.add_scalar('val/NLL', loss.item(), validation_step)
        tbx.add_scalar('val/lossPerEpoch', torch.mean(torch.tensor(losses)).item(), epoch)

        if len(args.gpu_ids) > 0:
            torch.cuda.empty_cache()

        if (epoch + 1) > 20:
            checkpoint_path = os.path.join('save', 'model-eur-akde-' + str(epoch + 1) + '.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print("model saved to {}".format(checkpoint_path))


if __name__ == '__main__':
    main(sys.argv[1:])
