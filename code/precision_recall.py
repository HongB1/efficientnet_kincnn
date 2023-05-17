
import torch

def precision_recall(y_true, y_pred):
    '''Calculate F1 score. Can work with gpu tensors
    
    The original implmentation is written by Michal Haltuf on Kaggle.
    
    Returns
    -------
    torch.Tensor
        `ndim` == 1. epsilon <= val <= 1
    
    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    - http://www.ryanzhang.info/python/writing-your-own-loss-function-module-for-pytorch/
    '''

    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
    
    epsilon = 1e-7
    
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    
    f1 = 2 * (precision*recall) / (precision + recall + epsilon)

    return precision, recall, f1



# def epoch_record(fold, mode, loss_, dataset_sizes, precision_, recall_, f1_, corrects_):
    
#     if mode == 'train': 
#         epoch_train_loss = loss_ / dataset_sizes[f'train']
#         epoch_train_precision = precision_ / dataset_sizes[f'train']
#         epoch_train_recall = recall_ / dataset_sizes[f'train']
#         epoch_train_f1 = f1_ / dataset_sizes[f'train']
#         epoch_train_acc = corrects_ / dataset_sizes[f'train']
#         print(f'''train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} 
#             F1: {epoch_train_f1:.4f} Precision: {epoch_train_precision:.4f} Recall: {epoch_train_recall:.4f}''')
            
#         return epoch_train_loss, epoch_train_acc, epoch_train_f1, epoch_train_precision, epoch_train_recall

#     else:
#         epoch_val_loss = loss_ / dataset_sizes[f'valid']
#         epoch_val_precision = precision_ / dataset_sizes[f'valid']
#         epoch_val_recall = recall_ / dataset_sizes[f'valid']
#         epoch_val_f1 = f1_ / dataset_sizes[f'valid']
#         epoch_val_acc = corrects_ / dataset_sizes[f'valid']
#         globals()[f'{fold}_result'].append(epoch_val_loss)
#         print(f'''valid Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f} 
#             F1: {epoch_val_f1:.4f} Precision: {epoch_val_precision:.4f} Recall: {epoch_val_recall:.4f}''')

#         return epoch_train_loss, epoch_val_loss, epoch_train_acc, epoch_val_acc, epoch_train_f1, epoch_val_f1, epoch_train_precision, epoch_val_precision, epoch_train_recall, epoch_val_recall


# def epoch_best_record(loss_, dataset_sizes, precision_, recall_, f1_, corrects_):
#     epoch_train_loss = loss_ / dataset_sizes
#     epoch_train_precision = precision_ / dataset_sizes
#     epoch_train_recall = recall_ / dataset_sizes
#     epoch_train_f1 = f1_ / dataset_sizes
#     epoch_train_acc = corrects_ / dataset_sizes
#     print(f'''train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} 
#     F1: {epoch_train_f1:.4f} Precision: {epoch_train_precision:.4f} Recall: {epoch_train_recall:.4f}''')

    
#     return epoch_train_loss, epoch_train_acc, epoch_train_f1, epoch_train_precision,  epoch_train_recall



# # epoch마다 아래 정보를 출력
#     writer.add_scalars('Loss' , {f'train_{fold}':epoch_train_loss, f'validation_{fold}':epoch_val_loss}, epoch)
#     writer.add_scalars('Accuracy' , {f'train_{fold}':epoch_train_acc, f'validation_{fold}':epoch_val_acc},  epoch)
#     writer.add_scalars('F1' , {f'train_{fold}':epoch_train_f1, f'validation_{fold}':epoch_val_f1},  epoch)
#     writer.add_scalars('precision' , {f'train_{fold}':epoch_train_precision, f'validation_{fold}':epoch_val_precision},  epoch)
#     writer.add_scalars('recall' , {f'train_{fold}':epoch_train_recall, f'validation_{fold}':epoch_val_recall},  epoch)
