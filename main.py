
from utils import best_path_improved, Lextree, State
import hmm
#import pydot
import numpy as np
from tqdm import tqdm
import librosa
import segk
from matplotlib import pyplot as plt
from segk import HMM
from mfcc_s import get_MFCC
import os
import SGMM
import pickle
from sgmm_p2 import find_true_seq
def save_model(models, file_name):
    with open(file_name, 'wb') as file:
        pickle.dump(models, file)


def load_model(filename):
    with open(filename, 'rb') as fr:
        model = pickle.load(fr)
    return model
def extract_recognition_result(sequence):
    out=''
    flag=True
    for char in sequence:

        if isinstance(char,State):
          #  print(char.name)
            if char.name.startswith('r') or char.name.startswith('s'):
                continue
            elif flag:
                flag=False
                out+=char.name[2]
        elif char=='NE':
            flag=True
    return out

def split_MFCC(mfcc, sequence_label,data):

    i=0
    for st in sequence_label[1:]:
        if not isinstance(st,State):
            continue
        elif st.name.startswith('silence'):
            i+=1
            continue
        else:
            if st.name in data.keys():
                data[st.name].append(mfcc[i,:])
            else:
                data[st.name]=[mfcc[i,:]]
            i+=1
    # for name, dts in data.items():
    #     means[name]=np.mean(np.array(dts),axis=0)
    #     vars[name]=np.vars(np.array(dts),axis=0)
    #     print(means['0'].shape)
   # print(data['0_state_0'])
   #  for k, v in data.items():
   #      print(f"{k}:{np.array(v).shape}")
    return data
  #  return means, vars
def train():

    flag=True


    cost=np.inf
    #HMM_list={1: hmm, 2:hmm}
    ## INITIALIZE
    HMM_list = SGMM.build_shmm()
    silence_data = []
    file = f"template/silence"
    for wav_name in os.listdir(file):
        if wav_name.endswith(".wav"):
            f = os.path.join(f"template/silence", wav_name)
            silence_data.append(get_MFCC(f))
    silence_HMM = SGMM.train_model_sghmm(silence_data, 2)
    lex_tree = Lextree(HMM_list,silence_HMM=silence_HMM,silence=True)
    lex_tree.create_states_for_digits()
   # lex_tree = load_model('lextree_ol.pkl')

    save_model(lex_tree,'lex_tree_origin.pkl')
    ## TRAIN
    costs=[]
    for i in tqdm(range(15)):

        new_data={}
        new_cost=0
        n=0
        means,vars={},{}
        for i,samples in enumerate(os.listdir("template_conitnue")):

            if samples.endswith('.wav'):
                name= samples[:-4].split('_')[0]
              #  print('name',name)

                lex_tree.create_states_for_digits(name)

                n=i
                audio_path = f'template_conitnue/{samples}'
                mfccs=get_MFCC(audio_path)
                # 使用

                best_path_result = best_path_improved(mfccs,LexTree=lex_tree, pruning_val=1.005)
              #  print('path:',extract_recognition_result(best_path_result.sequence))
                new_cost+=best_path_result.dtw

                seq=best_path_result.sequence
                new_data= split_MFCC(mfcc=mfccs,sequence_label=seq,data=new_data)
        for name, dts in new_data.items():
            means[name]=np.mean(np.array(dts),axis=0)
            vars[name]=np.var(np.array(dts),axis=0)

        for name, data in means.items():
            lex_tree.trained_hmms[name[0]].means[int(name[-1])]=data
        for name, data in vars.items():
            lex_tree.trained_hmms[name[0]].var[int(name[-1])] = data

        #save_model(lex_tree, f'lex_tree{int(new_cost/n)}.pkl')
        costs.append(new_cost/n)
      #  print('cur_err:', new_cost/n)
        if not ((cost-new_cost>(2*n)) or (new_cost>cost)):
            break
        cost=new_cost
    lex_tree.create_states_for_digits()
    save_model(lex_tree,'lex_tree.pkl')
#
    plt.plot(costs)
    plt.show()
train()
def Test():

    tree= load_model('lex_tree_final.pkl')
    audio_path = 'template_conitnue\\0123456789.wav'
    mfccs = get_MFCC(audio_path)

    best_path_result = best_path_improved(mfccs, tree, pruning_val=1.005)
    print(extract_recognition_result(best_path_result.sequence))
# 1.Path.sequence str -> list of states
# 2.构建 lextree 由定标签决定
# 3.best path -> {7-1: [[1,39], [1,39],...], }
def wer(reference, hypothesis):
    ref = list(reference)
    hyp = list(hypothesis)

    d = np.zeros((len(ref) + 1) * (len(hyp) + 1), dtype=np.uint8).reshape((len(ref) + 1, len(hyp) + 1))

    # Initialize the matrix for Levenshtein distance
    for i in range(len(ref) + 1):
        d[i][0] = i
    for j in range(len(hyp) + 1):
        d[0][j] = j

    # Compute the Levenshtein distance
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                cost = 0
            else:
                cost = 1
            d[i][j] = min(d[i - 1][j] + 1,  # deletion
                          d[i][j - 1] + 1,  # insertion
                          d[i - 1][j - 1] + cost)  # substitution

    # Calculate WER
    wer_value = d[len(ref)][len(hyp)] / float(len(ref))
    return wer_value
tree= load_model('lex_tree.pkl')
total_wer=0
nums=0
for audio_path in os.listdir('.\\'):
    if audio_path.endswith('.wav'):
        nums+=1
        reference = audio_path.split('.')[0]
        mfccs = get_MFCC(audio_path)

        best_path_result = best_path_improved(mfccs, tree, pruning_val=1.005)
        hypothesis = extract_recognition_result(best_path_result.sequence)
        W = wer(reference,hypothesis)
        total_wer+=W
        print (reference)
        print(hypothesis)
        print(f"WER: {W}")
print("Average_WER:", total_wer/nums)