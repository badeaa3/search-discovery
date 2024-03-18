#! /usr/bin/env python

'''
Author: Anthony Badea
Date: March 18, 2023
'''

# python packages
import os
import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

import h5py
import datetime
import json
# import yaml

# custom code
from batcher import loadData
from model import Model

if __name__ == "__main__":

    # user options
    parser = argparse.ArgumentParser(usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument("-c", "--config_file", help="Configuration file.", default="./config_files/default_config.json")
    # parser.add_argument("-i", "--inFile", help="Input training file.", default=None, required=True)
    parser.add_argument("-o", "--outDir", help="File name of the output directory", default="./checkpoints")
    parser.add_argument("-e", "--max_epochs", help="Max number of epochs to train on", default=None, type=int)
    parser.add_argument("-s", "--max_steps", help="Max number of steps to train on", default=-1, type=int)
    parser.add_argument("-d", "--device", help="Device to use.", default=None)
    parser.add_argument("-w", "--weights", help="Initial weights.", default=None)
    ops = parser.parse_args()

    # # load configuration
    # print(f"Using configuration file: {ops.config_file}")
    # with open(ops.config_file, 'r') as fp:
    #     config = json.load(fp)
    # print(config)

    # decide on device
    device = ops.device
    if not device:
        device = "gpu" if torch.cuda.is_available() else "cpu"
    pin_memory = (device == "gpu")

    # load data
    X, Y = loadData()
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.1)
    print(f"X_train {X_train.shape}, Y_train {Y_train.shape}, X_val {X_val.shape}, Y_val {Y_val.shape}")
    train_dataloader = DataLoader(TensorDataset(X_train, Y_train), shuffle=True, num_workers=4) # pin_memory=pin_memory) #, batch_size=config["batch_size"])
    val_dataloader = DataLoader(TensorDataset(X_val, Y_val), shuffle=False, num_workers=4) #, pin_memory=pin_memory) #, batch_size=config["batch_size"])

    # make checkpoint dir
    checkpoint_dir = os.path.join(ops.outDir, f'training_{datetime.datetime.now().strftime("%Y.%m.%d.%H.%M.%S")}')
    print(f"Saving checkpoints to: {checkpoint_dir}")
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # create model
    model = Model([X.shape[1], 10, 5], False, [5, 3, 1], False) #**config["model"])
    #model = torch.compile(model) # ready for pytorch 2.0 once it's more stable

    # callbacks
    callbacks = [
        ModelCheckpoint(monitor="train_loss", dirpath=checkpoint_dir, filename='cp-{epoch:04d}-{step}', every_n_train_steps = 1, save_top_k=20), # 0=no models, -1=all models, N=n models, set save_top_k=-1 to save all checkpoints
    ]

    # torch lightning trainer
    trainer = pl.Trainer(
        accelerator=device,
        devices=1,
        max_epochs=ops.max_epochs,
        max_steps=ops.max_steps,
        log_every_n_steps=5,
        callbacks=callbacks,
        default_root_dir=checkpoint_dir,
        #detect_anomaly=True,
        #**config["trainer"]
    )
    
    # fit
    trainer.fit(model, train_dataloader, val_dataloader)
    
    # save model
    trainer.save_checkpoint(os.path.join(checkpoint_dir,"finalWeights.ckpt"))
