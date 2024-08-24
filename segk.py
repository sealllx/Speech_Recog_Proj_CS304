import numpy as np
from scipy.stats import multivariate_normal
import os
from mfcc_s import get_MFCC

class state:
    def __init__(self,means,vars,rank,is_non=False,is_silence=False,number=None):
        self.number=number
        self.means=means
        self.vars=vars
        self.is_non=is_non
        self.is_silence=is_silence
        self.children=[]
        self.rank=rank
        self.transition=0.05 # not with log

    def gaussian(self,mfcc):
        return np.sum(np.log((np.exp((-(mfcc-self.means)**2)/(2*np.sqrt(self.vars))))/np.sqrt(2*np.pi*self.vars)))
        #得到高斯概率
class Path:
    def __init__(self, sequence,node,left_seq, dtw=0):
        self.sequence = sequence
        self.left_seq=left_seq
        self.dtw = dtw
        self.node=node
class HMM:
    def __init__(self, n_states, n_features, label):
        self.label=label
        # Hidden Markov Model initialization
        self.n_states = n_states  # Number of hidden states
        self.n_features = n_features  # Number of observation features
        # Initialize transition probability matrix
        self.transition_probs = np.full((n_states, n_states), 1.0 / n_states)
        # Initialize Gaussian emission probabilities with mean and variance
        self.means = np.random.rand(n_states, n_features)
        self.vars = np.random.rand(n_states, n_features) + 1.0  # Ensure positive variance
        # Initialize initial state probabilities
        self.initial_probs = np.full(n_states, 1.0 / n_states)

    def train(self, training_data):
        # Train the model using segmental K-means algorithm
        states=self.segmental_k_means(training_data)
        return states
    def gaussian_prob2(self, x, state):
        # Calculate Gaussian probability (multivariate)
        mean = self.means[state]
        cov = self.vars[state]
        distribution = multivariate_normal(mean=mean, cov=cov)
        pdf_value = distribution.pdf(x)
        return pdf_value

    def gaussian_prob(self, x, state):
        # Calculate Gaussian probability density function
        mean = self.means[state]
        var = self.vars[state]
        small_const = 1e-6
        var = np.clip(var, small_const, None)  # Prevent division by zero
        coeff = 1 / np.sqrt(2 * np.pi * var)
        exponent = np.exp(-0.5 * ((x - mean) ** 2) / var)
        prob = coeff * exponent
        return prob.prod()

    def segmental_k_means(self, Data, max_iter=100):
        # Segmental K-means algorithm

        for data in Data:
            n_samples, n_features = data.shape
            assert n_features == self.n_features  # Ensure correct feature dimension

            # Initialize state assignments uniformly
            states = np.floor(np.linspace(0, self.n_states, n_samples, endpoint=False)).astype(int)

            for iteration in range(max_iter):
                new_means = np.zeros_like(self.means)
                new_vars = np.zeros_like(self.vars)

                # Update means and variances for each state
                for s in range(self.n_states):

                    assigned_data = data[states == s, :]
                    if assigned_data.size == 0: continue  # Skip if no data points for this state

                    new_means[s] = assigned_data.mean(axis=0)
                    new_vars[s, :] = assigned_data.var(axis=0)
                    self.transition_probs[s, s] = (len(assigned_data) - 1) / len(assigned_data)
                    try:
                        self.transition_probs[s,s+1]=1-self.transition_probs[s,s]
                    except:
                        pass
                # Check for convergence

                if np.allclose(self.means, new_means, atol=1e-7):
                    break

                # Update model parameters
                self.means = new_means
                self.vars = new_vars

                # Reassign states based on updated parameters
                for i in range(n_samples):
                    states[i] = np.argmax([np.log(self.gaussian_prob(data[i, :], k)) for k in range(self.n_states)])
        all_states=[]
        for i in range(self.n_states):
            all_states.append(state(means=self.means[i], vars=self.vars[i], rank=i, number=self.label))
            if len(all_states) > 1:
                all_states[-2].children.append(all_states[-1])
                all_states[-1].transition=self.transition_probs[i-1,i]
      #  all_states[-1].children.append(state(means=None, vars=None,rank=0, is_non=True))

        return all_states, self.transition_probs


class Tree:
    def __init__(self):
        self.root=state(means=None,vars=None,rank=0, is_non=True)
        self.Data={}
        for i in range(16):  # template from 1 to 5
            file = f"template/template_{i}"
            for wav_name in os.listdir(file):
                if wav_name.endswith(".wav"):
                    f = os.path.join(f"template/template_{i}", wav_name)
                    if wav_name[:-4] in self.Data.keys():
                        self.Data[wav_name[:-4]].append(get_MFCC(f))
                    else:
                        self.Data[wav_name[:-4]] = [get_MFCC(f)]
    def build(self,added_data=None):
        # Step1: load data
        statelist_1={}  #for 0-9
        statelist_2={}  #for 2-9
        statelist_3={}  #for silence

        if added_data:
            for lb, data in added_data.items():
                if lb in self.Data.keys():
                    self.Data[lb].append(data)
                else:
                    self.Data[lb]=[data]

        # silence_data=[]
        # file="silence"
        # for wav_name in os.listdir(file):
        #     if wav_name.endswith(".wav"):
        #         silence_data.append(get_MFCC(wav_name))

        # Step2: HMM & Segmental Kmeans
        # HMM 每个数字的不同state之间构建父子关系, 用segmental k-means得到（transition prob, gaussian means & variance）
        HMMs={}
        for label,data in self.Data.items():
            HMMs[label]=HMM(n_states=5, n_features=39, label=label)
            states=HMMs[label].train(data)
            statelist_1[label]=states
            if int(label)>1:
                statelist_2[label]=states

        return HMMs

lextree = Tree()
HMM_list = lextree.build()
print("2")






