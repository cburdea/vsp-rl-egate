
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
    print('l_max anfragen')
    print('c1 anfragen')
    print('c2 anfragen')
    print('unterschied N-Steps und Rollout Steps nachfragen: Warum nicht gleich?')
    print('Terminologie hin zu VSP')
    print('Forken zu VSP')
    print('Warum wird im Training eine Batchzahl größer 1 benötigt?')
    print('Probleme nochmal neu berechnen, falls parameter angepasst werden müssen')

    args = args()

    device = torch.device(args.device)
    ROLLOUT_STEPS = int(args.ROLLOUT_STEPS)
    N_ROLLOUT = int(args.N_ROLLOUT)
    N_EPOCHS = int(args.N_EPOCHS)

    #reader.save_plans_as_pickle("100")


    model = Model(input_node_dim=8, hidden_node_dim=64, input_edge_dim=2, hidden_edge_dim=16)
    model.to(device)
    net = torch.nn.DataParallel(model, device_ids=[0,1], output_device=device)

    #model.load_state_dict(torch.load("model/v8-tw-iter200-rm25-latest_custom.model"))


    train(model=model,
          epochs=N_EPOCHS,
          n_rollout=N_ROLLOUT,
          rollout_steps=ROLLOUT_STEPS,
          train_steps=1,
          n_remove=10,
          )


    # train(model, envs, 1000, 20, 10, 4, n_remove=10)

    #torch.save(model.state_dict(), "model/v8-tw-iter200-rm25-latest_custom.model")


    gc.collect()
