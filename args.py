import argparse


def get_preprocessing_args(args):

    parser = argparse.ArgumentParser()
    add_common_args(parser)

    parser.add_argument('--input_data_location', type=str, default="./", help='Location where input data files are stored.')

    return parser.parse_args(args)


def get_cap_args(args):
    parser = argparse.ArgumentParser()
    add_common_args(parser)

    parser.add_argument('--tr_batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--start_from', type=str, default='', help='')

    parser.add_argument('--learning_rate_decay_start', type=int, default=0,
                        help='at what iteration to start decaying learning rate? (-1 = dont) (in epoch)')
    parser.add_argument('--learning_rate_decay_every', type=int, default=20,
                        help='how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.95,
                        help='how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--dataset', type=str, default='eurlex_raw_text.p',
                        help='Options: eurlex_raw_text.p, rcv1_raw_text.p, wiki30k_raw_text.p')
    parser.add_argument('--vocab_size', type=int, default=30001, help='vocabulary size')

    parser.add_argument('--agent_length', type=int, default=300, help='the length of agent text.')
    parser.add_argument('--cust_length', type=int, default=200, help='the length of customer text.')
    parser.add_argument('--is_AKDE', type=bool, default=True, help='if Adaptive KDE routing is enabled')
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for training')
    parser.add_argument('--num_compressed_capsule', type=int, default=128, help='The number of compact capsules')
    parser.add_argument('--dim_capsule', type=int, default=16, help='The number of dimensions for capsules')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of sub-processes to use per data loader.')
    parser.add_argument('--save_dir', type=str, default='./save/', help='Base directory for saving information.')
    parser.add_argument('--stop_time', type=float, default=0.0, help='Whether to be stopped in limited hours so that could be run on kaggle.com.')
    return parser.parse_args(args)


def add_common_args(parser):
    parser.add_argument('--vec_size', type=int, default=50, help='embedding size')
    parser.add_argument('--preprocessed_data_location', type=str, default="./data/", help='Location where output data files will be stored.')
