import timeit
start = timeit.default_timer()

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

    args = args()

    device = torch.device(args.device)
    ROLLOUT_STEPS = int(args.ROLLOUT_STEPS)
    N_ROLLOUT = int(args.N_ROLLOUT)

    reader.save_plans_as_pickle("100")

    stop = timeit.default_timer()
    print('Data loaded: ', stop - start)

    model = Model(input_node_dim=8, hidden_node_dim=64, input_edge_dim=2, hidden_edge_dim=16)
    model = model.to(device)

    #model.load_state_dict(torch.load("model/v8-tw-iter200-rm25-latest_custom.model"))

    '''
    train(model=model,
          epochs=5,
          n_rollout=N_ROLLOUT,
          rollout_steps=ROLLOUT_STEPS,
          train_steps=4,
          n_remove=10,
          )
    '''

    # train(model, envs, 1000, 20, 10, 4, n_remove=10)

    #torch.save(model.state_dict(), "model/v8-tw-iter200-rm25-latest_custom.model")

    stop = timeit.default_timer()
    print('Time: ', (stop - start)/60)
    gc.collect()