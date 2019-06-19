from __future__ import division, print_function, unicode_literals
from args import get_cap_args
import torch
import torch.nn as nn
import os
import json
import time
from torch.optim import Adam
from network import CapsNet_Text, BCE_loss
from w2v import load_word2vec
import sys
from utils import set_seeds, get_available_devices
from tqdm import tqdm
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from dataset_classes import dynamicProcessing
"""
TODO: writing an evaluation script for X_test using load_model.
TODO: saving the best model along with last 3-4(variable on args) models.
TODO: is it OK to set seeds by calling a method in another file?
TODO: is it necessary to call torch.cuda.empty_cache() that often?
"""


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


def transformLabels(labels, device):
    label_index = list(set([l for _ in labels for l in _]))
    label_index.sort()

    variable_num_classes = len(label_index)
    batch_size = len(labels)
    my_target = torch.zeros((batch_size, variable_num_classes), dtype=torch.float, device=device)
    for i_row, aRow in enumerate(my_target):
        aRow[[label_index.index(l) for l in labels[i_row]]] = 1
    # or equivalently:
    # my_target.scatter_(dim=-1, index=torch.tensor(list(map(lambda label: list(map(label_index.index, label)), labels))), value=1)
    assert torch.sum(my_target) != torch.tensor(0, dtype=torch.float, device=device)
    return torch.tensor(label_index, device=device), my_target


def main(main_args):
    set_seeds(0)
    args = get_cap_args(main_args)
    params = vars(args)
    print(json.dumps(params, indent=2))

    validation_step, train_step = 0, 0
    tbx = SummaryWriter(args.save_dir)

    the_dataset = dynamicProcessing(args)
    dataset_loader = torch.utils.data.DataLoader(the_dataset,
                                                 batch_size=args.tr_batch_size,
                                                 shuffle=True,
                                                 num_workers=args.num_workers,
                                                 collate_fn=my_collate_fn)

    embedding_weights = load_word2vec('QNB', embedding_dir=args.preprocessed_data_location)
    args.num_classes = the_dataset.get_class_count()

    device, args.gpu_ids = get_available_devices()
    args.tr_batch_size *= max(1, len(args.gpu_ids))
    model = CapsNet_Text(args, embedding_weights)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, args.gpu_ids)
    model = model.to(device)

    def scheduler_func(epoch):
        if epoch > args.learning_rate_decay_start and args.learning_rate_decay_start >= 0:
            frac = (epoch - args.learning_rate_decay_start) // args.learning_rate_decay_every
            decay_factor = args.learning_rate_decay_rate ** frac
        else:
            decay_factor = 1
        return decay_factor

    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scheduler_func)

    if args.stop_time:
        start_time = time.time()

    for epoch in range(1, args.num_epochs + 1):
        if args.stop_time and is_better_to_stop(epoch, start_time, args.stop_time):
            break
        torch.cuda.empty_cache()

        scheduler.step()

        the_dataset.set_train()
        model.train()

        with torch.enable_grad(), tqdm(total=len(dataset_loader.dataset)) as progress_bar:
            for X, Y in dataset_loader:
                start = time.time()

                batch_labels, batch_target = transformLabels(Y, device)
                data = torch.stack(X, dim=0).to(device)
                optimizer.zero_grad()
                poses, activations = model(data, batch_labels)
                loss = BCE_loss(activations, batch_target)
                loss.backward()
                optimizer.step()
                torch.cuda.empty_cache()

                progress_bar.update(args.tr_batch_size)
                progress_bar.set_postfix(epoch=epoch, NLL=loss.item(), elapsed=time.time() - start)
                train_step += args.tr_batch_size
                tbx.add_scalar('train/NLL', loss.item(), train_step)

        torch.cuda.empty_cache()
        the_dataset.set_val()
        model.eval()
        losses = []

        with torch.no_grad(), tqdm(total=len(dataset_loader.dataset)) as progress_bar:
            for X, Y in dataset_loader:
                start = time.time()

                data = torch.stack(X, dim=0).to(device)
                poses, activations = model(data, None)
                Y = torch.tensor(Y, device=device, dtype=torch.float)
                loss = BCE_loss(activations, Y)
                losses.append(loss)

                progress_bar.update(args.tr_batch_size)
                progress_bar.set_postfix(epoch=epoch, NLL=loss.item(), elapsed=time.time() - start)
                validation_step += args.tr_batch_size
                tbx.add_scalar('val/NLL', loss.item(), validation_step)
        tbx.add_scalar('val/lossPerEpoch', torch.mean(torch.tensor(losses)).item(), epoch)

    checkpoint_path = os.path.join('save', 'model-eur-akde-' + 'last' + '.pth')
    torch.save(model.state_dict(), checkpoint_path)
    print("model saved to {}".format(checkpoint_path))


if __name__ == '__main__':
    main(sys.argv[1:])
