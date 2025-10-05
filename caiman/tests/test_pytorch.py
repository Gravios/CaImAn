#!/usr/bin/env python

import numpy as np
import os

from caiman.paths import caiman_datadir
from caiman.pytorch_model_arch import PyTorchCNN, keras_cnn_model_from_pickle

try:
    os.environ["KERAS_BACKEND"] = "torch"
    try:
        import keras_core as keras
    except ImportError:
        import keras
    use_keras = True
except(ModuleNotFoundError):
    import torch 
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset 
    use_keras = False

def test_torch():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    try:
        model_name = os.path.join(caiman_datadir(), 'model', 'cnn_model')
        if use_keras:
            import pickle
            model_file = model_name + ".pkl"
            with open(model_file, 'rb') as f:
                print('USING MODEL:' + model_file)
                pickle_data = pickle.load(f)
                
            loaded_model = keras_cnn_model_from_pickle(pickle_data, keras)
        else:
            model_file = model_name + ".pt"
            model = PyTorchCNN()
            model.load_state_dict(torch.load(model_file))
    except:
        raise Exception(f'NN model could not be loaded.')

    try:
        if use_keras:
            dataset = np.random.randn(10, 50, 50, 1)        
            predictions = loaded_model.predict(dataset, batch_size=32)
        else:
            dataset = TensorDataset(torch.randn(10, 1, 50, 50))
            loader = DataLoader(dataset, batch_size=32) 
            
            model.eval()
            with torch.no_grad():
                for batch in loader:
                    predictions = model(batch[0])
        pass 
    except:
        raise Exception('NN model could not be deployed.')

if __name__ == "__main__":
    test_torch()
