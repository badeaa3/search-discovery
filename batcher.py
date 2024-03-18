'''
Author: Anthony Badea
Date: March 18, 2023
'''

import pandas as pd
import numpy as np
import torch

def loadData(inFile):
    
    df = pd.read_hdf(inFile) # "data/testfile_files100_35.h5"
    n = 100
    # inputs
    x = [i for i in df.columns if "201" not in i and "isinSR" not in i and "model" not in i]
    x = torch.Tensor(np.array(df[:n][x]))
    y = [i for i in df.columns if "201" in i]
    y = torch.Tensor(np.array(df[:n][y]))
    return x, y

if __name__ == "__main__":
    
    X, Y = loadData()
    print(X.shape, Y.shape)

    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, TensorDataset

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.1)
    print(f"X_train {X_train.shape}, Y_train {Y_train.shape}, X_val {X_val.shape}, Y_val {Y_val.shape}")
    train_dataloader = DataLoader(TensorDataset(X_train, Y_train), shuffle=True, num_workers=4) # pin_memory=pin_memory) #, batch_size=config["batch_size"])
    val_dataloader = DataLoader(TensorDataset(X_val, Y_val), shuffle=False, num_workers=4) #, pin_memory=pin_memory) #, batch_size=config["batch_size"])