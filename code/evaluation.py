import torch

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
import numpy as np
import sklearn
import os 
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


def predict_model(model_name, df, matrix_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 1024
    matrix = []
    for i in df[matrix_name]:
        matrix.append(i)

    matrix = np.array(matrix).astype('float64')
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

def ensemble_5fold(model_list, df, matrix_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predict_list = []
    for model in model_list:
        prediction = predict_model(model_name=model, df=df, matrix_name=matrix_name)
        predict_list.append(prediction)
    ensemble = (predict_list[0] + predict_list[1] + predict_list[2] + predict_list[3] + predict_list[4])/len(predict_list)
    return ensemble

def ensemble(modelname, df):
    ensemble = ensemble_5fold(modelname, df, 'matrix')

    df['probs'] = ensemble
    if 'answer' in df.columns :
        df['answer'] = df['answer'].apply(lambda x: int(x)) 
        auc = sklearn.metrics.roc_auc_score(df['answer'], df[f'probs'])
        return auc