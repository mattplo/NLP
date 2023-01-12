import numpy as np
import pandas as pd
from numba import int64, njit


def accuracy(y, y_hat):
    return np.count_nonzero(y == y_hat)/y.shape[0]


def random_model(X, nb_pos, rng):
    return rng.choice(nb_pos, size=X.shape)


def random_model_wdist(X, nb_pos, pos_freq, rng):
    return rng.choice(nb_pos, size=X.shape, p=pos_freq)


def model_most_common_pos(X, pos_freq):
    most_common_pos = np.argmax(pos_freq)
    return np.full(X.shape, most_common_pos)


@njit
def rand_choice_numba(arr, prob):
    """
    :param arr: A 1D numpy array of values to sample from.
    :param prob: A 1D numpy array of probabilities for the given samples.
    :return: A random sample from the given array with a given probability.
    """
    return arr[np.searchsorted(np.cumsum(prob), np.random.random(), side="right")]


@njit
def model_dist_type(X, types_to_pos_dist, seed=1):
    nb_tokens = X.shape[0]
    nb_pos = types_to_pos_dist.shape[1]
    np.random.seed(seed)
    pred = np.zeros_like(X)
    for i in range(nb_tokens):
        pos_freq = types_to_pos_dist[X[i]]
        pos = np.arange(nb_pos)
        pred[i] = rand_choice_numba(pos, pos_freq)
    return pred


def model_most_common_type(X, types_to_pos_dist):
    most_common_pos = np.argmax(types_to_pos_dist, axis=1)
    pred = most_common_pos[X]
    return pred


@njit
def viterbi_algo(A, B, pi, sentence):
    # A of shape nb_pos x nb_pos
    # B of shape nb_pos x nb_types
    N = A.shape[0]
    T = sentence.shape[0]

    viterbi = np.zeros((N, T))
    backpointer = np.zeros((N, T))
    pred = np.zeros(T, dtype=int64)

    for s in range(N):
        viterbi[s, 0] = pi[s]*B[s, sentence[0]]
        backpointer[s, 0] = 0

    for t in range(1, T):
        for s in range(N):
            for sp in range(N):
                val = viterbi[sp, t-1]*A[sp, s]*B[s, sentence[t]]
                if val > viterbi[s, t]:
                    viterbi[s, t] = val
                    backpointer[s, t] = sp

    bestpathpointer = np.argmax(viterbi[:, -1])
    pred[-1] = bestpathpointer
    for t in range(T-2, -1, -1):
        pred[t] = backpointer[pred[t+1], t+1]

    return pred


@njit
def model_viterbi(X, start_sentences, len_sentences, A, B, pi):
    pred = np.zeros(X.shape, dtype=int64)
    nb_sentences = start_sentences.shape[0]
    for i in range(nb_sentences):
        start = start_sentences[i]
        end = start+len_sentences[i]
        sentence = X[start:end]
        pred[start:end] = viterbi_algo(A, B, pi, sentence)
    return pred
