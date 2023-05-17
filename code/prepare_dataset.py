import sys

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data_utils
from sklearn.model_selection import train_test_split


def prepare_dataset(dataset_mode=None):
    df = pd.read_pickle("/home/hb/python/phospho/data/required/0308_final_train_test/train1.pkl")
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    # df['stratify'] 컬럼의 unique value가 1인 값들은 df_로 따로 빼고, 나머지는 df로
    # df['stratify'].value_counts() : df['stratify'] 컬럼의 unique value들을 count해주는 함수
    # [df['stratify'].value_counts()==1].index.tolist() : df['stratify'] 컬럼에서 unique value가 1인 index만 리스트로 추출
    df["stratify"] = df.apply(lambda x: x["kin_id"] + "-" + str(x["answer"]), axis=1)
    df_ = df[
        df["stratify"].isin(
            df["stratify"]
            .value_counts()[df["stratify"].value_counts() == 1]
            .index.tolist()
        )
    ]
    df = df[
        ~df["stratify"].isin(
            df["stratify"]
            .value_counts()[df["stratify"].value_counts() == 1]
            .index.tolist()
        )
    ]

    # df['matrix']는 pandas.series type이므로 list형태로 바꿔준다.
    matrix = np.array([x for x in df["matrix"]], dtype=np.float32)

    # answer에 df['answer']의 값을 list형태로 넣어준다
    answer = list(df["answer"].astype("int"))

    # 아까 따로 뗴어놨던 df_에 대해서도 똑같이 matrix_를 만어준다.
    matrix_ = np.array([x for x in df_["matrix"]], dtype=np.float32)
    answer_ = list(df_["answer"].astype("int"))

    # matrix = np.concatenate([matrix, matrix_])
    # answer = np.concatenate([answer, answer_])

    # dataset = data_utils.TensorDataset(torch.tensor(matrix), torch.tensor(answer))
    # return dataset
    # answer = np.array(answer, dtype=np.int64)
    # answer_ = np.array(answer_, dtype=np.int64)
    # print(matrix.shape, answer.shape)
    xTrain, xTest, yTrain, yTest = train_test_split(
        matrix, answer, test_size=0.2, random_state=42, stratify=df["stratify"]
    )
    try:
        xTrain = np.concatenate([xTrain, matrix_])
        yTrain = np.concatenate([yTrain, answer_])
    except:
        pass
    # print(type(yTrain[0]))
    print(yTrain)
    counts = np.bincount(yTrain)
    print(
        "Number of positive samples in training data: {} ({:.2f}% of total)".format(
            counts[1], 100 * float(counts[1]) / len(yTrain)
        )
    )
    counts = np.bincount(yTest)
    print(
        "Number of positive samples in validation data: {} ({:.2f}% of total)".format(
            counts[1], 100 * float(counts[1]) / len(yTest)
        )
    )  # counts[1] : answer = 1 인 값
    # globals()['counts'] = counts
    # BATCH_SIZE = batch_size
    train_set = data_utils.TensorDataset(torch.tensor(xTrain), torch.tensor(yTrain))
    valid_set = data_utils.TensorDataset(torch.tensor(xTest), torch.tensor(yTest))

    return train_set, valid_set

    # elif dataset_mode == 'pretrain_nonrepl':
    #     # df_nonreplicated = pd.read_pickle("/home/hb/python/phospho/data/atlas/ats_nonreplicated_with_psph.pkl")
    #     # # with open("/home/hb/python/phospho/data/atlas/matrix_0595.pickle", 'rb') as f:
    #     #     # matrix = pickle.load(f)
    #     # # with open("/home/hb/python/phospho/data/atlas/label_0595.pickle", 'rb') as f:
    #     #     # label = pickle.load(f)
    #     # matrix = df_nonreplicated['matrix'].to_list()
    #     # matrix = np.array([x for x in matrix], dtype=np.float32)
    #     # answer = df_nonreplicated['answer'].to_numpy().astype(int)
    #     with open("/home/hb/python/phospho/data/atlas/0213_ats0595_mtx_nonrepl_with_psph.pkl", 'rb') as f:
    #         matrix = pickle.load(f)
    #     print('matrix loaded!')
    #     with open("/home/hb/python/phospho/data/atlas/0213_ats0595_ans_nonrepl_with_psph.pkl", 'rb') as f:
    #         label = pickle.load(f)
    #     print('label loaded!')
    #     xTrain, xTest, yTrain, yTest = train_test_split(matrix,
    #                                                     label,
    #                                                     test_size=0.2,
    #                                                     random_state=42,)

    #     train_set = data_utils.TensorDataset(
    #                             torch.tensor(xTrain), torch.tensor(yTrain))
    #     valid_set = data_utils.TensorDataset(
    #                             torch.tensor(xTest), torch.tensor(yTest))
    #     print('train_set loaded! :',len(train_set))
    #     print('valid_set loaded! :', len(valid_set))

    # elif dataset_mode == 'pretrain_1090':
    #     with open("/home/hb/python/phospho/data/atlas/1090/matrix1090.pkl", 'rb') as f:
    #         matrix = pickle.load(f)
    #     print('matrix loaded!')
    #     with open("/home/hb/python/phospho/data/atlas/1090/label1090.pkl", 'rb') as f:
    #         label = pickle.load(f)
    #     print('label loaded!')
    #     xTrain, xTest, yTrain, yTest = train_test_split(matrix,
    #                                                     label,
    #                                                     test_size=0.2,
    #                                                     random_state=42,)

    #     train_set = data_utils.TensorDataset(
    #                             torch.tensor(xTrain), torch.tensor(yTrain))
    #     valid_set = data_utils.TensorDataset(
    #                             torch.tensor(xTest), torch.tensor(yTest))
    #     print('train_set loaded! :',len(train_set))
    #     print('valid_set loaded! :', len(valid_set))

    # return train_set, valid_set


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


config = AttrDict()
