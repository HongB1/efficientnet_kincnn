#!/bin/bash

# python /home/hb/neoantigen/code/DeepNeo_.py 0 HLA-A 9 Natural log 512 500 1e-3 5 7 0.5 25 


# python /home/hb/neoantigen/code/DeepNeo_.py 0 HLA-B 9 Natural log 512 500 1e-3 5 7 0.5 25 


# python /home/hb/neoantigen/code/DeepNeo_.py 0 HLA-C 9 Natural log 512 500 1e-3 5 7 0.5 25 


# python /home/hb/neoantigen/code/DeepNeo_.py 0 HLA-A 9 Natural Normalize 512 500 1e-3 5 7 0.5 25 

# python /home/hb/neoantigen/code/DeepNeo_.py 0 HLA-A 10 Natural Normalize 512 500 1e-3 5 7 0.5 25

# python /home/hb/neoantigen/code/DeepNeo_.py 0 HLA-C 9 Natural Normalize 512 500 1e-3df_test_st 5 7 0.5 25 
#     config.gpu_num = sys.argv[1]
#     config.batch_size = int(sys.argv[2])
#     config.n_epoch = int(sys.argv[3])
#     config.defalut_learning_rate = float(sys.argv[4])
#     config.fold_num = int(sys.argv[5])
#     config.scheduler_patience, config.scheduler_factor = int(sys.argv[6]), float(sys.argv[7])
#     config.erls_patience = int(sys.argv[8])
#     config.dataset = sys.argv[9]
#     config.pretrain_fold_num = sys.argv[10]
#     config.model = f'efficientnet-phospho-B-15'
#     config.save_dir = f'/home/hb/python/efficientnet_kincnn/saved_model/{datetime.today().strftime("%m%d")}/DeepPP_{config.dataset}_{datetime.today().strftime("%H%M")}_bs{config.batch_size}_weight{config.pretrain_fold_num}'
python3 /home/hb/python/efficientnet_kincnn/DeepPhospho.py 1 1024 500 1e-3 5 7 0.7 50 kincnn2 0


