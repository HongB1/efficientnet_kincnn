from multiprocessing import Manager, Pool
import time
import os
import re
import numpy as np
import pandas as pd
import multiprocessing
from collections import OrderedDict
import parmap
import torch
from tqdm import tqdm

class LoadingModel():
    """
    Model 불러오는 클래스 만들기
    """
    # filedir_list = []

    def __init__(self, _args):
        self._path = _args.path
        self._filedir_list = []
        self._bs = _args.batch_size
    def __repr__(self):
        return 'Model Saved path : {}'.format(self._path)
        
    def set_saved_dir(self):
        os.chdir(self._path)
        self._filedir_list = []
        self._filedir_list = [f'{self._path}/{i}' for i in os.listdir()]
        self._filedir_list.sort()
        return self._filedir_list

    def loading_model(self):
        for idx, filedir in enumerate(self._filedir_list):
            filedir_found = re.search('HLA-(.+?)_bs', filedir).group(1)
            print(filedir_found)
            globals()[f'model_list_{filedir_found}_bs{self._bs}'] = load_multiple_model(filedir, filedir_found)

def load_hlaseq():

    hla_a_prot = pd.read_csv('../data/need/HLA_A_prot.txt', sep='\t', header = None)
    hla_b_prot = pd.read_csv('../data/need/HLA_B_prot.txt', sep='\t', header = None)
    hla_c_prot = pd.read_csv('../data/need/HLA_C_prot.txt', sep='\t', header = None)
    hla_prot = pd.concat([hla_a_prot, hla_b_prot, hla_c_prot], axis = 0)

    hla = {}
    for line in hla_prot.to_numpy():
        hla[line[0]] = line[1]
        
    return hla
    
def parallelize_dataframe(df, func, num_cores):
    df_split = np.array_split(df, num_cores) # 코어 수 만큼 데이터 프레임 쪼개기
    data = parmap.map(func, df_split, pm_pbar = True, pm_processes=num_cores)
    try:
        data = pd.concat(data, ignore_index = True)
        return data
    except:
        return data
    

def make_matrix(df):
    # 해시값 불러오기
    hash_data = pd.read_csv('../data/need/Calpha.txt', sep='\t')
    hash_data.set_index(hash_data['Unnamed: 0'], inplace=True)
    # 불필요한 칼럼 정리
    del hash_data['Unnamed: 0']
    hash_data = np.exp(-1*hash_data)
       
    hash_data_list = []
    #hla = load_hlaseq()
    
    for amino in hla[df['allele']]:
        if amino in hash_data.columns:
            for target in df['Peptide seq']:
                if target == '*' or target == 'X' or target == 'U':
                    hash_data_list.append(0)
                else:
                    hash_data_list.append(hash_data[amino][target])
        else:
            for target in df['Peptide seq']:
                hash_data_list.append(0)
    
    if 'HLA-A' in df['allele'] and len(df['Peptide seq'])==9:
        matrix = np.array(hash_data_list).reshape(1,276,9).astype('float32')
    elif 'HLA-B' in df['allele'] and len(df['Peptide seq'])==9:
        matrix = np.array(hash_data_list).reshape(1,252,9).astype('float32')
    elif 'HLA-C' in df['allele'] and len(df['Peptide seq'])==9:
        matrix = np.array(hash_data_list).reshape(1,268,9).astype('float32')
    elif 'HLA-A' in df['allele'] and len(df['Peptide seq'])==10:
        matrix = np.array(hash_data_list).reshape(1,276,10).astype('float32')
    elif 'HLA-B' in df['allele'] and len(df['Peptide seq'])==10:
        matrix = np.array(hash_data_list).reshape(1,252,10).astype('float32')
    elif 'HLA-C' in df['allele'] and len(df['Peptide seq'])==10:
        matrix = np.array(hash_data_list).reshape(1,268,10).astype('float32')
    return matrix


def make_df(df):
    df['matrix'] = df.apply(make_matrix_short_hla, axis = 1)
    return df


def use_multicore(df, num_cores = multiprocessing.cpu_count()):
    df = parallelize_dataframe(df, make_df, num_cores)
    return df



def load_short_hlaseq():

    hla_a_prot = pd.read_csv('../HLA_A_prot.txt', sep='\t', header = None)
    hla_b_prot = pd.read_csv('../HLA_B_prot.txt', sep='\t', header = None)
    hla_c_prot = pd.read_csv('../HLA_C_prot.txt', sep='\t', header = None)
    
    hla_a_prot[1] = hla_a_prot[1].map(lambda x: x[24:-65])
    hla_c_prot[1] = hla_c_prot[1].map(lambda x: x[4:-66])
    hla_b_prot[1] = hla_b_prot[1].map(lambda x: x[12:-62])
    
    hla_prot = pd.concat([hla_a_prot, hla_b_prot, hla_c_prot], axis = 0)
    hla = {}
    for line in hla_prot.to_numpy():
        hla[line[0]] = line[1]
        
    return hla


def load_hlaseq():

    hla_a_prot = pd.read_csv('../data/need/HLA_A_prot.txt', sep='\t', header = None)
    hla_b_prot = pd.read_csv('../data/need/HLA_B_prot.txt', sep='\t', header = None)
    hla_c_prot = pd.read_csv('../data/need/HLA_C_prot.txt', sep='\t', header = None)
    hla_prot = pd.concat([hla_a_prot, hla_b_prot, hla_c_prot], axis = 0)

    hla = {}
    for line in hla_prot.to_numpy():
        hla[line[0]] = line[1]
        
    return hla


def make_matrix_short_hla(df):
    # 해시값 불러오기
    hash_data = pd.read_csv('../data/need/Calpha.txt', sep='\t')
    hash_data.set_index(hash_data['Unnamed: 0'], inplace=True)
    # 불필요한 칼럼 정리
    del hash_data['Unnamed: 0']
    hash_data = np.exp(-1*hash_data)
       
    hash_data_list = []
    hla = load_short_hlaseq()
    
    for amino in hla[df['allele']]:
        if amino in hash_data.columns:
            for target in df['Peptide seq']:
                if target == '*' or target == 'X' or target == 'U':
                    hash_data_list.append(0)
                else:
                    hash_data_list.append(hash_data[amino][target])
        else:
            for target in df['Peptide seq']:
                hash_data_list.append(0)
    
    if 'HLA-A' in df['allele'] and len(df['Peptide seq'])==9:
        matrix = np.array(hash_data_list).reshape(1,276,9).astype('float32')
    elif 'HLA-B' in df['allele'] and len(df['Peptide seq'])==9:
        matrix = np.array(hash_data_list).reshape(1,252,9).astype('float32')
    elif 'HLA-C' in df['allele'] and len(df['Peptide seq'])==9:
        matrix = np.array(hash_data_list).reshape(1,268,9).astype('float32')
    elif 'HLA-A' in df['allele'] and len(df['Peptide seq'])==10:
        matrix = np.array(hash_data_list).reshape(1,276,10).astype('float32')
    elif 'HLA-B' in df['allele'] and len(df['Peptide seq'])==10:
        matrix = np.array(hash_data_list).reshape(1,252,10).astype('float32')
    elif 'HLA-C' in df['allele'] and len(df['Peptide seq'])==10:
        matrix = np.array(hash_data_list).reshape(1,268,10).astype('float32')
    
    return matrix


# 학습시켜놓은 모델을 불러오는 함수 정의
def load_multiple_model(filedir, datatype):
    model_path_list = [f'{filedir}/{file}' for file in os.listdir(filedir) if "best" in file]
    print(model_path_list)
    globals()['model_list_' + datatype] = []
    # estimators = []
    for i in range(len(model_path_list)):
        checkpoint = torch.load(model_path_list[i], map_location='cpu')
        globals()[datatype + '_fold' + str(i)] = checkpoint['model']
        globals()[datatype + '_fold' + str(i)].load_state_dict(checkpoint['state_dict'])
        globals()['model_list_'+datatype].append(globals()[datatype + '_fold' + str(i)])
    
    return globals()['model_list_'+datatype]

def predict(model_name, df):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 1024
    matrix = []
    for i in df['matrix']:
        matrix.append(i)

    matrix = np.array(matrix).astype('float32')
    dataset = torch.utils.data.TensorDataset(torch.tensor(matrix))
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)

    model = model_name.to(device)
    model.eval()
    preds_list = []
    probs_list = []
    with torch.no_grad():
        for matrix in tqdm(test_loader):
            matrix = torch.as_tensor(matrix[0], device=device, dtype=torch.float32)
            preds = model(matrix)
            probs_list.extend(preds.cpu().tolist())
            preds = torch.softmax(preds, dim=1)
            preds_list.extend(preds.cpu().tolist())

    return np.array(probs_list)

# 예측값을 ensemble하는 함수 정의
def ensemble_5fold(model_list, df):
    predict_list = []
    for model in model_list:
        prediction = predict(model_name=model, df=df)
        predict_list.append(prediction)
    ensemble = (predict_list[0] + predict_list[1] + predict_list[2] + predict_list[3] + predict_list[4])/len(predict_list)

    return ensemble



# test dataset을 만드는 함수 정의
def prepare_test_dataset(df, dataset_mode):
    df['length'] = df['Peptide_seq'].map(lambda x: len(x))
    df_A = df[df['allele'].str.contains('HLA-A')]
    df_B = df[df['allele'].str.contains('HLA-B')]
    df_C = df[df['allele'].str.contains('HLA-C')]

    globals()[dataset_mode + '_A_9'] = df_A[df_A['length']==9]
    globals()[dataset_mode + '_A_10'] = df_A[df_A['length']==10]
    globals()[dataset_mode + '_B_9'] = df_B[df_B['length']==9]
    globals()[dataset_mode + '_B_10']= df_B[df_B['length']==10]
    globals()[dataset_mode + '_C_9'] = df_C[df_C['length']==9]
    globals()[dataset_mode + '_C_10'] = df_C[df_C['length']==10]
    
    return globals()[dataset_mode + '_A_9'], globals()[dataset_mode + '_A_10'], globals()[dataset_mode + '_B_9'], globals()[dataset_mode + '_B_10'], globals()[dataset_mode + '_C_9'], globals()[dataset_mode + '_C_10']

def peptide_seq(df):
    df_answer1 = df[df['answer']==1]
    df_answer1_len9 = df_answer1[df_answer1['length']==9]
    df_answer1_len10 = df_answer1[df_answer1['length']==10]
    
    peptide_len9 = list(df_answer1_len9['Peptide seq'])
    peptide_len10 = list(df_answer1_len10['Peptide seq'])

    return peptide_len9, peptide_len10


class LoadingModel():
    """
    Model 불러오는 클래스 만들기
    """
    # filedir_list = []

    def __init__(self, _args):
        self._path = _args.path
        self._filedir_list = []
        self._bs = _args.batch_size
    def __repr__(self):
        return 'Model Saved path : {}'.format(self._path)
        
    def set_saved_dir(self):
        os.chdir(self._path)
        self._filedir_list = []
        self._filedir_list = [f'{self._path}/{i}' for i in os.listdir()]
        self._filedir_list.sort()
        return self._filedir_list

    def loading_model(self):
        for idx, filedir in enumerate(self._filedir_list):
            filedir_found = re.search('HLA-(.+?)_bs', filedir).group(1)
            print(filedir_found)
            globals()[f'model_list_{filedir_found}_bs{self._bs}'] = load_multiple_model(filedir, filedir_found)

        return globals()[f'model_list_{filedir_found}_bs{self._bs}'] 

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


