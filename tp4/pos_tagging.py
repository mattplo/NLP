import numpy as np
import pandas as pd
# from sklearn.model_selection import dataset_test_split
from sklearn.decomposition import PCA
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Perceptron
from src.Dataset import Dataset


def generate_dict_features(dataset, features):
    X = [{"word": dataset.types[dataset.X[i]],
          "left_neighbor":dataset.types[dataset.X[i-1]],
          "right_neighbor":dataset.types[dataset.X[(i+1) % dataset.nb_tokens]],
          "pos_in_sentence":0,
          "end_word3":dataset.types[dataset.X[i]][-3:],
          "end_word2":dataset.types[dataset.X[i]][-2:],
          "word_length":len(dataset.types[dataset.X[i]]),
          "word_length_left":len(dataset.types[dataset.X[i-1]]),
          "word_length_right":len(dataset.types[dataset.X[(i+1) % dataset.nb_tokens]]),
          "left_pos":"",
          "after_comma":0,
          "before_comma":0
          } for i in range(dataset.nb_tokens)]

    for i in range(dataset.nb_sentences):
        X[dataset.start_sentences[i]]["left_neighbor"] = ""
        X[dataset.start_sentences[i]+dataset.len_sentences[i]-1]["right_neighbor"] = ""
        X[dataset.start_sentences[i]]["word_length_left"] = 0
        X[dataset.start_sentences[i] +
            dataset.len_sentences[i]-1]["word_length_right"] = 0
        for j in range(dataset.len_sentences[i]):
            s = dataset.start_sentences[i]
            if dataset.len_sentences[i] > 1:
                X[s+j]["pos_in_sentence"] = (j/(dataset.len_sentences[i]-1))
            X[s+j]["left_end_word3"] = X[s+j]["left_neighbor"][-3:]
            X[s+j]["left_end_word2"] = X[s+j]["left_neighbor"][-2:]
            X[s+j]["right_end_word3"] = X[s+j]["right_neighbor"][-3:]
            X[s+j]["right_end_word2"] = X[s+j]["right_neighbor"][-2:]
            if X[s+j]["left_neighbor"] != "":
                X[s+j]["left_pos"] = dataset.pos[dataset.y[s+j-1]]
            X[s+j]["after_comma"] = (X[s+j]["left_neighbor"] == ",")
            X[s+j]["before_comma"] = (X[s+j]["right_neighbor"] == ",")

    X_filtered = [{key: X[i][key] for key in features}
                  for i in range(dataset.nb_tokens)]
    return X_filtered


def eval_perceptron(train, test, features, limit_n=False, do_pca=False, pca_ncomponents=1000):

    if not limit_n:
        limit_n = train.nb_tokens

    X_train = generate_dict_features(train, features)
    X_test = generate_dict_features(test, features)

    v = DictVectorizer(sparse=(not do_pca))
    v.fit(X_train[:limit_n])
    X_onehot_train = v.transform(X_train[:limit_n])
    X_onehot_test = v.transform(X_test[:limit_n])

    if do_pca:
        pca = PCA(n_components=pca_ncomponents)
        X_matrix = pca.fit_transform(X_matrix)

    Y_train = np.array([train.pos[train.y[i]] for i in range(train.nb_tokens)])
    Y_test = np.array([test.pos[test.y[i]] for i in range(test.nb_tokens)])

    print(X_onehot_train.shape)
    print(X_onehot_test.shape)

    clf = Perceptron(random_state=0)
    clf.fit(X_onehot_train, Y_train)

    return clf.score(X_onehot_test, Y_test)


# train_languages = ["fr_gsd-ud-train.conllu",
    #    "et_edt-ud-train.conllu", "fi_tdt-ud-train.conllu", "en_ewt-ud-train.conllu"]
train_languages = ["en_ewt-ud-train.conllu"]
test_languages = ["et_edt-ud-dev.conllu", "fi_tdt-ud-dev.conllu"]
for train_language in train_languages:
    for test_language in test_languages:
        train = Dataset(train_language)
        test = Dataset(test_language)
        score = eval_perceptron(train, test, [
                                "word", "left_neighbor", "right_neighbor", "pos_in_sentence", "end_word3", "end_word2", "word_length"])
        print("train:", train_language, "|test:",
              test_language, "|score:", score)

# train: fr_gsd-ud-train.conllu |test: et_edt-ud-dev.conllu |score: 0.24083757933315456
# train: fr_gsd-ud-train.conllu |test: fi_tdt-ud-dev.conllu |score: 0.22858859514966134
# train: et_edt-ud-train.conllu |test: et_edt-ud-dev.conllu |score: 0.8959506570126039
# train: et_edt-ud-train.conllu |test: fi_tdt-ud-dev.conllu |score: 0.4544461437622897
# train: fi_tdt-ud-train.conllu |test: et_edt-ud-dev.conllu |score: 0.4630151068204166
# train: fi_tdt-ud-train.conllu |test: fi_tdt-ud-dev.conllu |score: 0.8842582477605418
# train: en_ewt-ud-train.conllu |test: et_edt-ud-dev.conllu |score: 0.31465093411996065
# train: en_ewt-ud-train.conllu |test: fi_tdt-ud-dev.conllu |score: 0.24202534411186366
