import numpy as np
import torch
import lib.input_reader as input_reader
from lib.utils_eval import create_batch_env, roll_out, random_init
from lib.egate_model import Model
from arguments import args
import os, sys
import csv
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)


if __name__ == "__main__":
    args = args()

    device = torch.device(args.device)

    N_JOBS = int(args.N_JOBS)
    batch_size = int(args.BATCH)
    n_steps=int(args.N_STEPS)
    eval_mode = str(args.EVAL_MODE)

    model = Model(input_node_dim=8, hidden_node_dim=64, input_edge_dim=2, hidden_edge_dim=16)
    model = model.to(device)
    model.load_state_dict(torch.load("/home/cb/PycharmProjects/masterarbeit_cpu/src/vsp/model/final_model_operational.model"))

    inputs = input_reader.load_vsp_envs_from_pickle("vsp_data_100/pickle_test_data/" + eval_mode)

    log = [["plan_nr", "Random Cost", "Before Cost", "After Cost", "History"]]

    for index, vsp_instance in enumerate(inputs):
        print('-----------------------------------------------------------------------------------------------------')
        print('Plan:', index)
        envs = create_batch_env(1, N_JOBS, vsp_instance)

        states, mean_cost = random_init(envs, 100, 1, N_JOBS)
        random_cost = mean_cost

        states = envs.reset()
        before_mean_cost = np.mean([env.cost for env in envs.envs])
        print ("before mean cost:", before_mean_cost)
        states,history, actions_history, values_history = roll_out(model,envs,states,n_steps)

        after_mean_cost = np.mean([env.cost for env in envs.envs])
        print ("after mean cost:", after_mean_cost)
        history = np.array(history)

        a = [env.env.tours() for env in envs.envs]

       # for i in range(len(a)):
       #     print(a[i])

        log.append([[index, random_cost, before_mean_cost, after_mean_cost, history]])

    log_path = parentdir + '/' + 'log_eval_' + eval_mode +'.csv'
    for l in log:
        with open(log_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(l)