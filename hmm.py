import os
import numpy as np
from hmmlearn import hmm
import librosa


# 提取MFCC特征
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return mfccs.T  # 转置是为了让时间序列在第一维度

# 准备训练数据
def prepare_training_data(base_dir='template'):
    training_data = {}
    for digit in range(10):
        training_data[digit] = []
        for template_num in range(1, 6):
            folder_name = f'template_{template_num}'
            file_path = os.path.join(base_dir, folder_name, f'{digit}.wav')
            mfcc_features = extract_features(file_path)
            training_data[digit].append(mfcc_features)
    return training_data

# 初始化HMM模型
# def init_hmm(n_states=3, n_features=13):
#     n_components = n_states + 1  # 加一个null state
#
#     model = hmm.GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=1000, init_params='', params='')
#
#     # 初始化状态转移矩阵
#     transmat = np.zeros((n_components, n_components))
#     transmat[:-1, :-1] = 0.5 * np.eye(n_states)  # 发射状态的自环和向前转移
#     transmat[:-1, -1] = 0.5  # 发射状态到非发射状态的转移
#     transmat[-1, -1] = 1.0  # 非发射状态的自环
#
#     # 初始化发射概率矩阵
#     means = np.zeros((n_components, n_features))
#     covars = np.ones((n_components, n_features))
#     # 为了简化，这里我们不对非发射状态的均值和方差做特殊处理
#     # 在实际应用中，您可能需要根据非发射状态的特性来调整这些值
#
#     startprob = np.zeros(n_components)
#     startprob[0] = 1.0  # 假设始终从第一个状态开始
#
#     # 分配参数到模型
#     model.startprob_ = startprob
#     model.transmat_ = transmat
#     model.means_ = means
#     model.covars_ = covars
#
#     return model


# def init_hmm(n_states=3):
#     # 创建HMM实例，这里使用GaussianHMM
#     model = hmm.GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=1000, verbose=True)
#     return model


def init_hmm(n_states=3, n_features=13):
    # 创建HMM实例，这里使用GaussianHMM
    model = hmm.GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=1000)

    # 初始化状态转移矩阵
    transmat = np.zeros((n_states, n_states))
    for i in range(n_states):
        if i < n_states - 1:
            transmat[i, i] = 0.5
            transmat[i, i + 1] = 0.5
        else:
            transmat[i, i] = 1.0  # 最后一个状态转移到自己

    # 初始化发射概率矩阵的均值和方差
    means = np.zeros((n_states, n_features))
    covars = np.ones((n_states, n_features))

    # 初始化开始概率
    startprob = np.zeros(n_states)
    startprob[0] = 1.0  # 假设始终从第一个状态开始

    # 分配参数到模型
    model.startprob_ = startprob
    model.transmat_ = transmat
    model.means_ = means
    model.covars_ = covars

    return model

# 训练模型
def train_models(training_data):
    models = {}
    for digit in training_data:
        model = init_hmm()
        X = np.concatenate(training_data[digit])
        lengths = [len(x) for x in training_data[digit]]
        print(f"Training HMM for digit {digit}")
        np.seterr(all='ignore')  # 忽略潜在的数值问题
        model.fit(X, lengths)  # 训练模型
        print(f"transmation {model.transmat_}")
        models[digit] = model
    return models


def evaluate_model(models, test_files):
    correct = 0
    total = 0
    for true_digit in test_files.keys():
        for test_file_path in test_files[true_digit]:
            # Start with a very low score for best_score
            best_score = float('-inf')
            predicted_digit = None

            for digit, model in models.items():
                mfcc_features = extract_features(test_file_path)
                score = model.score(mfcc_features)
                if score > best_score:
                    best_score = score
                    predicted_digit = digit

            if predicted_digit == true_digit:
                correct += 1
            total += 1

    return correct, total

def prepare_test_data(base_dir='test_set'):
    test_files = {}
    for digit in range(10):
        test_files[digit] = []
        file_path = os.path.join(base_dir, f'{digit}.wav')
        test_files[digit].append(file_path)
    return test_files

# 测试HMM模型
def test_models(models, base_test_dir='test_set'):
    test_data = prepare_test_data(base_test_dir)
    correct_digits, total_digits = evaluate_model(models, test_data)
    accuracy = correct_digits / total_digits if total_digits else 0
    print(f"Accuracy: {accuracy * 100:.2f}%")

# 主函数
def main():
    print("Extracting features and preparing training data...")
    training_data = prepare_training_data()
    print("Training HMM models for each digit...")
    models = train_models(training_data)
    print("Training complete!")
    return models

if __name__ == '__main__':
    trained_models = main()
    test_models(trained_models)
