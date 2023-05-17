import os
import sys

print(os.path.join(os.getcwd(), '..'))
sys.path.append(os.path.join(os.getcwd(), '..'))
import efficientnet_kincnn
import torch
model = efficientnet_kincnn.EfficientNet.from_name('efficientnet-phospho-B-15')