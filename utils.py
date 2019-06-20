import numpy as np
from multiprocessing import Pool
import random
import torch
from os.path import join
from pickle import load


def precision_at_k(r, k):
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)


def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    return np.sum(r / np.log2(np.arange(2, r.size + 2)))


def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max


def get_result(args):
    (y_pred, y_true) = args

    top_k = 50
    pred_topk_index = sorted(range(len(y_pred)), key=lambda i: y_pred[i], reverse=True)[:top_k]
    pos_index = set([k for k, v in enumerate(y_true) if v == 1])

    r = [1 if k in pos_index else 0 for k in pred_topk_index[:top_k]]

    p_1 = precision_at_k(r, 1)
    p_3 = precision_at_k(r, 3)
    p_5 = precision_at_k(r, 5)

    ndcg_1 = ndcg_at_k(r, 1)
    ndcg_3 = ndcg_at_k(r, 3)
    ndcg_5 = ndcg_at_k(r, 5)

    return np.array([p_1, p_3, p_5, ndcg_1, ndcg_3, ndcg_5])


def evaluate(Y_tst_pred, Y_tst):
    pool = Pool(12)
    results = pool.map(get_result, zip(list(Y_tst_pred), list(Y_tst)))
    pool.terminate()
    tst_result = list(np.mean(np.array(results), 0))
    print('\rTst Prec@1,3,5: ', tst_result[:3], ' Tst NDCG@1,3,5: ', tst_result[3:])


def set_seeds(anInt):
    torch.manual_seed(anInt)
    torch.cuda.manual_seed_all(anInt)
    np.random.seed(anInt)
    random.seed(anInt)


def get_available_devices():
    """Get IDs of all available GPUs.

    Returns:
        device (torch.device): Main device (GPU 0 or CPU).
        gpu_ids (list): List of IDs of all GPUs that are available.
    """
    gpu_ids = []
    if torch.cuda.is_available():
        gpu_ids += [gpu_id for gpu_id in range(torch.cuda.device_count())]
        device = torch.device('cuda:{}'.format(gpu_ids[0]))
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    return device, gpu_ids


def tempLoad(location, strname):
    with open(join(location, strname + ".pkl"), "rb") as fh:
        temp = load(fh)
    return torch.tensor(temp, dtype=torch.long)


def get_input_tensors(location, is_train=True):
    return (tempLoad(location, "X_train" if is_train else "X_val"),
            tempLoad(location, "y_train" if is_train else "y_val"))
