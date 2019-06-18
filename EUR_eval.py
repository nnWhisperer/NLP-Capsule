from __future__ import division, print_function, unicode_literals
from args import get_eval_args
import numpy as np
import torch
import torch.nn as nn
from network import CNN_KIM, CapsNet_Text
import time
from utils import evaluate, set_seeds, get_available_devices
import data_helpers
import scipy.sparse as sp
from w2v import load_word2vec
import os
import json
import sys
import tqdm


def load_model(model_name, args, embedding_weights, is_capsule=True):
    model = CapsNet_Text(args, embedding_weights) if is_capsule else CNN_KIM(args, embedding_weights)
    model.load_state_dict(torch.load(os.path.join(args.start_from, model_name)))
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, args.gpu_ids)
    print(model_name + ' loaded')
    return model


def main(main_args):
    set_seeds(0)
    a = torch.rand(5)
    print(a)
    # assert all(a == torch.tensor([0.4963, 0.7682, 0.0885, 0.1320, 0.3074]))
    args = get_eval_args(main_args)
    params = vars(args)
    print(json.dumps(params, indent=2))

    X_trn, Y_trn, Y_trn_o, X_tst, Y_tst, Y_tst_o, vocabulary, vocabulary_inv = data_helpers.load_data(args.dataset,
                                                                                                      max_length=args.sequence_length,
                                                                                                      vocab_size=args.vocab_size)
    Y_trn = Y_trn.toarray()
    Y_tst = Y_tst.toarray()

    X_trn = X_trn.astype(np.int32)
    X_tst = X_tst.astype(np.int32)
    Y_trn = Y_trn.astype(np.int32)
    Y_tst = Y_tst.astype(np.int32)

    embedding_weights = load_word2vec('glove', vocabulary_inv, args.vec_size)
    args.num_classes = Y_trn.shape[1]

    device, args.gpu_ids = get_available_devices()
    args.ts_batch_size *= max(1, len(args.gpu_ids))
    capsule_net = load_model('model-eur-akde-24.pth', args, embedding_weights)
    capsule_net = capsule_net.to(device)
    baseline = load_model('model-EUR-CNN-40.pth', args, embedding_weights, is_capsule=False)
    baseline = baseline.to(device)

    nr_tst_num = X_tst.shape[0]
    nr_batches = int(np.ceil(nr_tst_num / float(args.ts_batch_size)))

    n, k_trn = Y_trn.shape
    m, k_tst = Y_tst.shape
    print('k_trn:', k_trn)
    print('k_tst:', k_tst)

    capsule_net.eval()
    top_k = 50
    row_idx_list, col_idx_list, val_idx_list = [], [], []
    with tqdm(total=nr_batches) as progress_bar:
        for batch_idx in range(nr_batches):
            start = time.time()
            start_idx = batch_idx * args.ts_batch_size
            end_idx = min((batch_idx + 1) * args.ts_batch_size, nr_tst_num)
            X = X_tst[start_idx:end_idx]
            Y = Y_tst_o[start_idx:end_idx]
            data = torch.from_numpy(X).long().to(device)

            candidates = baseline(data)
            candidates = candidates.data.cpu().numpy()

            Y_pred = np.zeros([candidates.shape[0], args.num_classes])
            for i in range(candidates.shape[0]):
                candidate_labels = candidates[i, :].argsort()[-args.re_ranking:][::-1].tolist()
                _, activations_2nd = capsule_net(data[i, :].unsqueeze(0), candidate_labels)
                Y_pred[i, candidate_labels] = activations_2nd.squeeze(2).data.cpu().numpy()

            for i in range(Y_pred.shape[0]):
                sorted_idx = np.argpartition(-Y_pred[i, :], top_k)[:top_k]
                row_idx_list += [i + start_idx] * top_k
                col_idx_list += (sorted_idx).tolist()
                val_idx_list += Y_pred[i, sorted_idx].tolist()

            progress_bar.update(1)
            progress_bar.set_postfix(Reranking=args.re_ranking, elapsed=time.time() - start)

    m = max(row_idx_list) + 1
    n = max(k_trn, k_tst)
    Y_tst_pred = sp.csr_matrix((val_idx_list, (row_idx_list, col_idx_list)), shape=(m, n))

    if k_trn >= k_tst:
        Y_tst_pred = Y_tst_pred[:, :k_tst]

    evaluate(Y_tst_pred.toarray(), Y_tst)


if __name__ == '__main__':
    main(sys.argv[1:])
