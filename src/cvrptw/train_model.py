import timeit
start = timeit.default_timer()

import numpy as np
import json
import random
import math
import vrp_env
import torch
import gc
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data,DataLoader
from lib.utils_train import create_batch_env, train
from lib.egate_model import Model
from arguments import args
import lib.input_reader as vsp_env
import argparse

if __name__ == "__main__":
    args = args()

    device = torch.device('cpu')

    N_JOBS = int(args.N_JOBS)
    CAP = int(args.CAP)
    batch_size = int(args.BATCH)
    MAX_COORD = int(args.MAX_COORD)
    MAX_DIST = float(args.MAX_DIST)
    LR = float(args.LR)
    DEPOT_END = int(args.DEPOT_END)
    SERVICE_TIME = int(args.SERVICE_TIME)
    TW_WIDTH = int(args.TW_WIDTH)

    # envs = create_batch_env(8, 213, test=False)
    #envs = create_batch_env(8, 213, test=True)
    # envs = create_batch_env(1, 5, test=False)
    #print(envs)
    # envs = vsp_env.create_vsp_env_from_file("vsp_data/Fahrplan_213_1_1_L.txt")

    stop = timeit.default_timer()
    print('Data loaded: ', stop - start)


    model = Model(input_node_dim=8, hidden_node_dim=16, input_edge_dim=2, hidden_edge_dim=4)
    model = model.to(device)

    # model.load_state_dict(torch.load("model/v8-tw-iter200-rm25-latest.model"))
    train(model=model,envs=envs,epochs=1000,n_rollout=4 ,rollout_steps=2,train_steps=5,n_remove=213)
    #torch.save(model.state_dict(), "model/v8-tw-iter200-rm25-latest.model")

    stop = timeit.default_timer()
    print('Time: ', stop - start)
    gc.collect()