import argparse


def get_eval_args(args):

    parser = argparse.ArgumentParser()
    get_common_args(parser)

    parser.add_argument('--ts_batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--start_from', type=str, default='save', help='')

    parser.add_argument('--re_ranking', type=int, default=200, help='The number of re-ranking size')
    return parser.parse_args(args)


def get_cap_args(args):
    parser = argparse.ArgumentParser()
    get_common_args(parser)

    parser.add_argument('--tr_batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--start_from', type=str, default='', help='')

    parser.add_argument('--learning_rate_decay_start', type=int, default=0,
                        help='at what iteration to start decaying learning rate? (-1 = dont) (in epoch)')
    parser.add_argument('--learning_rate_decay_every', type=int, default=20,
                        help='how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.95,
                        help='how many iterations thereafter to drop LR?(in epoch)')
    return parser.parse_args(args)


def get_common_args(parser):
    parser.add_argument('--dataset', type=str, default='eurlex_raw_text.p',
                        help='Options: eurlex_raw_text.p, rcv1_raw_text.p, wiki30k_raw_text.p')
    parser.add_argument('--vocab_size', type=int, default=30001, help='vocabulary size')
    parser.add_argument('--vec_size', type=int, default=300, help='embedding size')
    parser.add_argument('--sequence_length', type=int, default=500, help='the length of documents')
    parser.add_argument('--is_AKDE', type=bool, default=True, help='if Adaptive KDE routing is enabled')
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for training')
    parser.add_argument('--num_compressed_capsule', type=int, default=128, help='The number of compact capsules')
    parser.add_argument('--dim_capsule', type=int, default=16, help='The number of dimensions for capsules')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of sub-processes to use per data loader.')
    parser.add_argument('--save_dir', type=str, default='./save/', help='Base directory for saving information.')
    parser.add_argument('--stop_time', type=float, default=0.0, help='Whether to be stopped in limited hours so that could be run on kaggle.com.')
