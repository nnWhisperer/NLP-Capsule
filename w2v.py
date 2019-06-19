from os.path import join
import pickle


def load_word2vec(model_type, embedding_dir="./data/"):
    if model_type == 'QNB':
        embedding_weights = None
        with open(join(embedding_dir, "embedding_weights.pkl"), "rb") as fh:
            embedding_weights = pickle.load(fh)
        return embedding_weights
    else:
        raise ValueError('Unknown pretrain model type: %s!' % (model_type))
