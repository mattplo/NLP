
import numpy as np


def parse(filename="fr_gsd-ud-train.conllu"):

    f = open('../data/{}'.format(filename), 'r', encoding="utf-8")

    sentences = []
    types_to_pos = []
    types = []
    token_to_type = {}
    nb_tokens = 0
    pos = []

    for line in f.readlines():
        if line[:8] == "# text =":
            sentences.append([])
        elif line[0].isdigit():

            split_line = line.split("\t")
            before_tab = split_line[0]

            if not "." in before_tab and not "-" in before_tab:
                token = split_line[1]
                pos_tag = split_line[3]
                if not token in token_to_type:
                    types.append(token)
                    token_to_type[token] = len(types)-1
                    types_to_pos.append([])
                if not pos_tag in pos:
                    pos.append(pos_tag)
                type_idx = token_to_type[token]
                pos_idx = pos.index(pos_tag)
                sentences[-1].append((type_idx, pos_idx))

                types_to_pos[type_idx].append(pos_idx)
                nb_tokens += 1

    max_words_in_sentence = 0
    for sentence in sentences:
        if len(sentence) > max_words_in_sentence:
            max_words_in_sentence = len(sentence)

    nb_sentences = len(sentences)

    types = np.array(types)
    pos = np.array(pos)
    nb_types = types.shape[0]
    nb_pos = pos.shape[0]

    len_sentences = np.zeros(nb_sentences, dtype=int)
    X = np.zeros(nb_tokens, dtype=int)
    y = np.zeros(nb_tokens, dtype=int)
    A = np.zeros((nb_pos, nb_pos))
    B = np.zeros((nb_pos, nb_types))

    idx = 0
    for (i, sentence) in enumerate(sentences):
        len_sentences[i] = len(sentence)
        for (j, (typ, pos_tag)) in enumerate(sentence):
            X[idx] = typ
            y[idx] = pos_tag
            if j > 0:
                A[y[idx-1], y[idx]] += 1
            B[pos_tag, typ] += 1
            idx += 1

    types_to_pos_dist = np.zeros((nb_types, nb_pos))
    for (i, pos_list) in enumerate(types_to_pos):
        for pos_tag in pos_list:
            types_to_pos_dist[i, pos_tag] += 1

    row_sums = np.sum(types_to_pos_dist, axis=1)
    types_to_pos_dist /= row_sums[:, np.newaxis]

    return X, y, types, pos, len_sentences, types_to_pos_dist, A, B


class Dataset:

    def __init__(self, filename="fr_gsd-ud-train.conllu") -> None:

        X, y, types, pos, len_sentences, types_to_pos_dist, A, B = parse(
            filename=filename)
        self.X = X  # vector of length self.nb_tokens
        self.y = y  # vector of length self.nb_tokens
        self.types = types  # vector of length self.nb_types
        self.pos = pos  # vector of length self.nb_pos
        self.len_sentences = len_sentences  # vector of length self.nb_sentences
        # matrix of size self.nb_types x self.nb_pos
        self.types_to_pos_dist = types_to_pos_dist

        # matrices A and B used in the Viterbi algorithm
        self.A = A
        self.B = B

        self.start_sentences = self.len_sentences.cumsum(
        ) - self.len_sentences  # vector of length nsentences

        self.nb_tokens = self.X.shape[0]
        self.nb_sentences = self.len_sentences.shape[0]
        self.nb_types = self.types.shape[0]
        self.nb_pos = self.pos.shape[0]

        # compute pos frequencies
        _, cnts = np.unique(self.y, return_counts=True)
        self.pos_freq = cnts/self.nb_tokens

        # normalize A and B
        self.A /= cnts[:, np.newaxis]
        self.B /= cnts[:, np.newaxis]

    def get_X_sentence(self, index):
        start = self.start_sentences[index]
        end = start+self.len_sentences[index]
        return self.X[start:end]

    def get_y_sentence(self, index):
        start = self.start_sentences[index]
        end = start+self.len_sentences[index]
        return self.y[start:end]

    def show_sentence(self, index):
        sentence = [self.types[i] for i in self.get_X_sentence(index)]
        pos = [self.pos[j] for j in self.get_y_sentence(index)]
        for (i, word) in enumerate(sentence):
            print(word, "({})".format(pos[i]), " ", end="", sep="")
