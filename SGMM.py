import numpy as np
import os
import librosa
from mfcc_s import get_MFCC

# Single gaussian with HMM
def log_gaussian_prob(x, means, var):
    return (-0.5 * np.log(var) - np.divide(np.square(x - means), 2 * var) - 0.5 * np.log(2 * np.pi)).sum()


class GSHMM():
    def __init__(self, single_wav, n_components):
        # assume every vec starts from the 1st state
        self.n_components = n_components
        self.init_prob = np.hstack((np.ones(1), np.zeros(n_components - 1)))
        self.means = np.tile(np.mean(single_wav, axis=0), (n_components, 1))

        self.var = np.tile(np.var(single_wav, axis=0), (n_components, 1))
       # print("sh:", self.var.shape)
        # store the state sequences that each vec went through
        self.states = []
        self.trans_mat = np.zeros((self.n_components, self.n_components))

    def k_seg(self, data):
        data_state = [[] for i in range(self.n_components)]
        # self.states stores the state that each vector belongs to
        # We classify all vec into different states in the data_state matrix
        for index, single_wav in enumerate(data):
            n = single_wav.shape[0]
            labeled_seq = self.states[index]
            for t, st in enumerate(labeled_seq[:-1]):
                self.trans_mat[st, labeled_seq[t + 1]] += 1
            for t in range(n):
                data_state[labeled_seq[t]].append(single_wav[t])
        data_state = [np.array(i) for i in data_state]
        # calculate new means and variance of the classified vec
        # print(data_state)
        for st in range(self.n_components):
            self.means[st, :] = np.mean(data_state[st], axis=0)
            self.var[st, :] = np.var(data_state[st], axis=0)
            self.trans_mat[st] /= np.sum(self.trans_mat[st])

    def viterbi(self, data):
        # print(self.means.shape)
        for index, single_wav in enumerate(data):

            n = single_wav.shape[0]
            labeled_seq = np.zeros(n, dtype=int)
            log_delta = np.zeros((n, self.n_components))
            psi = np.zeros((n, self.n_components))  # store the maximum path for each node in the graph
            log_delta[0] = np.log(self.init_prob)
            # initialize for the first
            for i in range(self.n_components):
                log_delta[0, i] += log_gaussian_prob(single_wav[0], self.means[i], self.var[i])
            log_trans_mat = np.log(self.trans_mat)
            # compute the viterbi path for each vec in a wav matrix
            for t in range(1, n):
                for st in range(self.n_components):
                    temp = np.zeros(self.n_components)
                    for i in range(self.n_components):
                        temp[i] = log_delta[t - 1, i] + log_trans_mat[i, st] + log_gaussian_prob(single_wav[t],
                                                                                                 self.means[st],
                                                                                                 self.var[st])
                    log_delta[t, st] = np.max(temp)
                    psi[t, st] = np.argmax(log_delta[t - 1] + log_trans_mat[:, st])
            labeled_seq[n - 1] = np.argmax(log_delta[n - 1])
            # reverse back for the optimal path
            for i in reversed(range(n - 1)):
                labeled_seq[i] = psi[i + 1, labeled_seq[i + 1]]
           # print('psi:',psi)
            self.states[index] = labeled_seq

    def forward(self, data):
        # forward algorithm for the max prob for a vec and a model
        alpha = np.zeros((data.shape[0], self.n_components))
        # initialize for the first vec
        alpha[0] = np.log(self.init_prob) + np.array(
            [log_gaussian_prob(data[0], self.means[j], self.var[j]) for j in range(self.n_components)])
        for i in range(1, data.shape[0]):
            for j in range(self.n_components):
                alpha[i, j] = log_gaussian_prob(data[i], self.means[j], self.var[j]) + np.max(
                    np.log(self.trans_mat[:, j].T) + alpha[i - 1])
        return alpha

    def train(self, data, iter):
        if iter == 0:
            # initialize optimal path seq
            for wav in data:
                seq = np.array([self.n_components * i / wav.shape[0] for i in range(wav.shape[0])], dtype=int)
                self.states.append(seq)
        self.k_seg(data)
        self.viterbi(data)

    def log_prob(self, data):
        alpha = self.forward(data)
        # choose the maximum path
        return np.max(alpha[data.shape[0] - 1])


def train_model_sghmm(data, n_components, max_iter=30):
    hmm_model = GSHMM(data[0], n_components=n_components)
    # start iteration
    iter = 0
    prev_prob = -np.infty
    log_prob = 0
    while iter <= max_iter:
        prev_prob = log_prob
        log_prob = 0.0
        hmm_model.train(data, iter)
        for wav in data:
            log_prob += hmm_model.log_prob(wav)
        # test if converge
        if abs(prev_prob - log_prob) < 0.1:
            break
        iter += 1

    return hmm_model


def evaluate_model2(test_dir, HMMs):
    correct = 0
    total = 0

    for wav_file in os.listdir(test_dir):
        if wav_file.endswith(".wav"):
            file_path = os.path.join(test_dir, wav_file)
            mfcc_features = get_MFCC(file_path)

            true_label = os.path.splitext(wav_file)[0]

            max_log_prob = -np.inf
            predicted_label = None
            for model_label, model in HMMs.items():
                log_prob = model.log_prob(mfcc_features)
                if log_prob > max_log_prob:
                    max_log_prob = log_prob
                    predicted_label = model_label

            if predicted_label == true_label:
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy


def build_shmm(added_data=None):
    Data = {}
    for i in range(15):  # template from 1 to 5
        file = f"template/template_{i}"
        for wav_name in os.listdir(file):
            if wav_name.endswith(".wav"):
                f = os.path.join(f"template/template_{i}", wav_name)
                if wav_name[:-4] in Data.keys():
                    Data[wav_name[:-4]].append(get_MFCC(f))
                else:
                    Data[wav_name[:-4]] = [get_MFCC(f)]

    if added_data:
        for lb, datas in added_data.items():
            for data in datas:
                if lb in Data.keys():
                    Data[lb].append(data)
                else:
                    Data[lb] = [data]


    SHMMs = {}
    for label, data in Data.items():
        SHMMs[label] = train_model_sghmm(data, n_components=5, max_iter=30)

    return SHMMs


SHMMs = build_shmm()
