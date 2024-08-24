import hmm
#import pydot
import numpy as np
import librosa
import segk
from segk import HMM
from mfcc_s import get_MFCC
import os


class State:
    def __init__(self, name, is_non_emitting=False, is_silence=False, means=None, variances=None):

        self.name = name
        self.number= self.name[2]
        self.is_non_emitting = is_non_emitting
        self.is_silence = is_silence
        self.means = means  # 添加均值
        self.variances = variances  # 添加方差
        self.transitions = []
        self.child_means=[]
        self.child_vars=[]
    def add_transition(self, next_state, probability=1.0):
        self.transitions.append((next_state, probability))
        if self.is_non_emitting:
            self.child_means.append(next_state.means)
            self.child_vars.append(next_state.variances)
class Path:
    def __init__(self, sequence,node,left_seq, dtw=0):
        self.sequence = sequence
        self.left_seq=left_seq
        self.dtw = dtw
        self.node=node
class Lextree:
    def __init__(self, trained_hmms):
        self.states = []
        self.trained_hmms = HMM_list
        self.silence_hmm= silence_HMM
        self.root = State("root_non_emitting", is_non_emitting=True)  # 根节点
        self.create_states_for_digits()

    def create_states_for_digits(self):
        # 初始化根非发射状态到数字状态的转移
        non_emitting_states = [self.root]  # 用于存储每轮的非发射状态

        for digit_round in range(8):  # 模拟最多7位数字
            new_non_emitting = State(f"none_emitting_{digit_round+1}", is_non_emitting=True, means = np.eye(39) * 0.0, variances = np.eye(39) * 1000.0)  # 为下一轮创建新的非发射状态
            self.states.append(new_non_emitting)

            for digit, hmm_model in self.trained_hmms.items():
                transition_matrix = hmm_model.transition_probs  # 应该使用当前hmm模型的转移概率矩阵
                # 对每个HMM模型提取状态信息
                prev_state = None
                for state_index in range(hmm_model.n_states):
                    state_name = f"{digit_round}_{digit}_state_{state_index}"
                    means = hmm_model.means[state_index]
                    variances = hmm_model.vars[state_index]
                    state = State(name=state_name, means=means, variances=variances)
                    self.states.append(state)
                    # 添加自循环
                    state.add_transition(state, probability=transition_matrix[state_index][state_index])
                    if prev_state:
                        prev_state.add_transition(state, probability=transition_matrix[state_index - 1][
                            state_index])  # 连接前一个状态到当前状态
                    else:
                        non_emitting_states[-1].add_transition(state, probability=1)  # 将数字的第一个状态连接到当前非发射状态
                        if digit_round>0 and add_silence:

                            non_emitting_states[-2].add_transition(state, probability=1) # 存在越过silence的可能性
                    prev_state = state

                # 数字的最后一个状态连接到新的非发射状态
                prev_state.add_transition(new_non_emitting, probability=1)  # 这里的概率可能需要根据你的模型进行调整

            non_emitting_states.append(new_non_emitting)  # 将这轮的非发射状态添加到列表中，以供下一轮使用
            if add_silence:

                new_non_emitting = State(f"none_emitting_sil_{digit_round + 1}", is_non_emitting=True,
                                                            means=np.eye(39) * 0.0, variances=np.eye(39) * 1000.0)
                transition_matrix=silence_HMM.transition_probs
                prev_state=None
                for state_index in range(silence_HMM.n_states):
                    state_name = f"silence_{state_index}"
                    means = silence_HMM.means[state_index]
                    variances = silence_HMM.vars[state_index]
                    state = State(name=state_name, means=means, variances=variances)
                   # self.states.append(state)
                    # 添加自循环
                    state.add_transition(state, probability=1)#transition_matrix[state_index][state_index])
                    if prev_state:
                        prev_state.add_transition(state, probability=1)#transition_matrix[state_index - 1][
                         #   state_index])  # 连接前一个状态到当前状态
                    else:
                        non_emitting_states[-1].add_transition(state, probability=1)  # 将数字的第一个状态连接到当前非发射状态

                    prev_state = state
                prev_state.add_transition(new_non_emitting,probability=1)
                non_emitting_states.append(new_non_emitting)  # 将这轮的非发射状态添加到列表中，以供下一轮使用

    def display_tree(self):
        # 简单显示树结构
        for state in self.states:
            transitions = ", ".join([f"{t[0].name} [prob={t[1]}]" if isinstance(t, tuple) else t.name for t in state.transitions])
            print(f"{state.name}: {transitions}")
def find_true_seq(sequence):
    splited= sequence.split("NE")
    out=''
    for num in splited:
        try:

            out=out+num[0]+''
        except:
          #  print('sequence: ',sequence)
            out=out+'N'
    return out
def best_path_improved(whole_sequence, LexTree, pruning_val):
    # 初始化路径列表，每个元素包含路径序列、当前节点、剩余序列和DTW成本
    paths = [Path(sequence='', node=LexTree.root, left_seq=whole_sequence, dtw=0)]
    min_err = float('inf')  # 用于存储最小的DTW成本，用于剪枝
    # 用于存储最终的最佳路径
    best_path = None
    length=whole_sequence.shape[0]

    while paths:
    #    threshes={}
        seq_cal={}
        count=0
        Passed=True
        new_paths = []
        # print(paths[0].sequence)
        flag=False
        for path in paths:

            if path.left_seq.size:
                flag=True
              #  if path.node.transitions[0][0].is_non_emitting:

            # 对于当前路径上的节点，遍历所有可能的转移
                for transition in path.node.transitions:
                    next_node, prob = transition
                    tunning_val= min([abs(path.left_seq.shape[0]-i* length/7) for i in range(1,7)])

                    if next_node.is_non_emitting:
                        count=0
                        new_dtw=path.dtw-max([calculate_gaussian_error2(means=next_node.child_means[i],vars=next_node.child_vars[i],
                                                               observation=path.left_seq[0]) for i in range(len(next_node.child_means))])*0.03-np.log(0.1)
                     #   print([calculate_gaussian_error2(means=next_node.child_means[i],vars=next_node.child_vars[i],
                            #                                   observation=path.left_seq[0]) for i in range(len(next_node.child_means))])
                        new_seq=path.sequence+'NE'
                        Passed= True



                    # 计算到下一个状态的高斯误差
                    else:
                        count+=1
                        if count>62:
                            Passed=False
                        error = calculate_gaussian_error(next_node, path.left_seq[0])*0.03
                        new_dtw = path.dtw - error - np.log(prob)  # 更新成本，考虑转移概率


                        # 如果新的DTW成本在允许的范围内，添加到新路径中
                       # if new_dtw < min_err * pruning_val:
                        new_seq = path.sequence + '' + next_node.number if path.sequence else next_node.number

                     #   new_paths.append(Path(sequence=new_seq, node=next_node, left_seq=path.left_seq[1:], dtw=new_dtw))
                    p=Path(sequence=new_seq, node=next_node, left_seq=path.left_seq[1:], dtw=new_dtw)
                    seq = find_true_seq(new_seq)[-1]
                    if seq not in seq_cal.keys():
                        seq_cal[seq] = [p]
                    else:
                        seq_cal[seq].append(p)
                    # if seq_cal[seq] < 200:
                    #     new_paths.append(
                    #         Path(sequence=new_seq, node=next_node, left_seq=path.left_seq[1:], dtw=new_dtw))
                  #  elif new_dtw<min()
                   # seq=find_true_seq(new_seq)
                  #  if seq not in threshes.keys():
                    #    threshes[seq]=[new_dtw]
                  #  else:
                     #   threshes[seq].append(new_dtw)
            else:

                continue
        if not flag:
            min_err = min([k.dtw for k in paths])
            for p in paths:
                if p.dtw==min_err:
                    return p

                        # min_error = new_dtw
        print("prob:",(prob))
        print("log_prob: ",np.log(prob))
        print("error:", error)
        #acc=len(new_paths)

       # thresh=sorted([k.dtw for k in new_paths])[:5000][-1]#min([k.dtw for k in new_paths])

    #    max_err=max([k.dtw for k in new_paths])


        paths=[]
        for seq,p_list in seq_cal.items():
           # for p in sorted(p_list, key=lambda i: i.dtw)[:200]:
            new_paths.append(sorted(p_list, key=lambda i: i.dtw)[0])
        thresh=sorted([k.dtw for k in new_paths])[:5000][-1]
        min_err = min([k.dtw for k in new_paths])
        print("min_err: ", min_err)
        for p in new_paths:

            if p.dtw<=thresh:#*pruning_val:
                paths.append(p)
            if p.dtw==min_err:
                best_path=p
                print("e.g. ",p.sequence)
                print("left: ",p.left_seq.shape[0])
        print("length of path: ", len(paths))

        print("possis: ",seq_cal.keys())



        # 更新路径列表
        #paths = new_paths

    return best_path


def calculate_gaussian_error2(means,vars, observation):
    means = means#np.array(node.means)
    # variances = np.array([node.variances[i][i] for i in range(len(node.variances))])
    variances = vars#np.array(node.variances)
    # print(type(variances))
    # print(len(variances))

    error = np.sum(np.log((np.exp((-(observation-means)**2)/(2*np.sqrt(variances))))/np.sqrt(2*np.pi*variances)+1e-10))
    # error = np.sum(
    #     (np.exp((-(observation - means) ** 2) / (2 * np.sqrt(variances)))) / np.sqrt(2 * np.pi * variances))
    # error = np.sum((observation - means) ** 2 / (2 * variances))
    return error



def calculate_gaussian_error(node, observation):
    means = np.array(node.means)
    # variances = np.array([node.variances[i][i] for i in range(len(node.variances))])
    variances = np.array(node.variances)
    # print(type(variances))
    # print(len(variances))

    error = np.sum(np.log((np.exp((-(observation-means)**2)/(2*np.sqrt(variances))))/np.sqrt(2*np.pi*variances)+1e-10))
    # error = np.sum(
    #     (np.exp((-(observation - means) ** 2) / (2 * np.sqrt(variances)))) / np.sqrt(2 * np.pi * variances))
    # error = np.sum((observation - means) ** 2 / (2 * variances))
    return error



audio_path = '5678995.wav'
# y, sr = librosa.load(audio_path, sr=None)
# mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
# mfccs = mfccs.T
mfccs=get_MFCC(audio_path)
print(mfccs.shape)

# 使用
lextree = segk.Tree()
HMM_list = lextree.build()
silence_HMM=HMM(n_states=2, n_features=39, label='sil')
add_silence=True
silence_data=[]
file = f"template/silence"
for wav_name in os.listdir(file):
    if wav_name.endswith(".wav"):
        f = os.path.join(f"template/silence", wav_name)
        silence_data.append(get_MFCC(f))
silence_HMM.train(silence_data)
lextree = Lextree(HMM_list)
# lextree.display_tree()
#visualize_tree_simplified(lextree.root)

best_path_result = best_path_improved(mfccs, lextree, pruning_val=1.005)
#best_path_result = best_path_improved(mfccs, lextree, pruning_val=1.005)

print("yes")
def extract_recognition_result(best_path):
    # 初始化识别结果字符串
    out=''
    for char in find_true_seq(best_path.sequence):
        if not char=='l':
            out=out+char

    return out

# 使用
recognition_result = extract_recognition_result(best_path_result)
print("识别结果:", recognition_result)