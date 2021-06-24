
import os, sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import numpy as np
import json
import random
import math
import vrp_env
import torch
import gc
import lib.input_reader as reader
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data,DataLoader
from lib.utils_train import create_batch_env, train
from lib.egate_model import Model
from arguments import args
import argparse

if __name__ == "__main__":
    args = args()

    device = torch.device(args.device)
    ROLLOUT_STEPS = int(args.ROLLOUT_STEPS)
    N_ROLLOUT = int(args.N_ROLLOUT)
    N_EPOCHS = int(args.N_EPOCHS)
    REMOVE_NUMBER = int(args.REMOVE_NUMBER)

    model = Model(input_node_dim=4, hidden_node_dim=64, input_edge_dim=3, hidden_edge_dim=16)
    #model = torch.nn.DataParallel(model)
    model.to(device)


    #model.load_state_dict(torch.load("model/v8-tw-iter200-rm25-latest_custom.model"))


    train(model=model,
          epochs=N_EPOCHS,
          n_rollout=N_ROLLOUT,
          rollout_steps=ROLLOUT_STEPS,
          train_steps=4,
          n_remove=REMOVE_NUMBER)


    gc.collect()
