import numpy as np
import torch
import lib.input_reader as input_reader
from lib.utils_eval import create_batch_env, roll_out, random_init
from lib.egate_model import Model
from arguments import args
from tabulate import tabulate
import os, sys
import csv
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

exact_solution=[
    86379.279296875,
    103591.439453125,
    103591.439453125,
    103591.439453125,
    83062.1195068359,
    124380.119140625,
    124380.119140625,
    124380.119140625,
    58677.6796875,
    55580.759765625,

    55580.759765625,
    55580.759765625,
    113613.19921875,
    79458.0390625,
    79458.0390625,
    79458.0390625,
    79810.798828125,
    88619.798828125,
    88619.798828125,

    88619.798828125,
    58632.0399169922,
    74439.6796875,
    74439.6796875,
    74439.6796875,
    81525.7990722656,
    64214.8791503906,
    64214.8791503906,
    64214.8791503906,
    71389.5592041016,
    87546.439453125,

    87546.439453125,
    87546.439453125,
    111039.39855957,
    104171.159179688,
    104171.159179688,
    104171.159179688,
    43311.0404052734,
    64437.958984375,
    64437.958984375,

    64437.958984375,
    98968.119140625,
    134317.798828125,
    108515.879272461,
    134776.280029297,
    86050.51953125,
    112714.03918457,
    125290.19921875,
    107716.679931641,
    98603.55859375,

    144053.318603516,
    74449.7197875977,
    92831.6398925781,
    91215.6389160156,
    120304.599609375,
    88554.2388916016,
    92392.87890625,
    142581.838134766,
    140966.478637695,
    52831.8004150391,

    124946.51953125,
    92034.71875,
    119304.958984375,
    92556.6796875,
    112554.439941406,
    72211.359375,
    80695.3594360352,
    144302.438476563,
    97810.6799316406,
    102233.039306641,

    114554.718994141,
    76812.0390625,
    71155.8798828125,
    101904.239624023,
    108594.599609375,
    115829.71875,
    84519.87890625,
    126190.119140625,
    106999.798950195,
    95225.359375,

    104315.599609375,
    104017.55859375,
    134317.798828125,
    112758.599609375,
    134776.280029297,
    104836.0390625,
    112714.03918457,
    158807.278320313,
    107716.679931641,
    128499.718994141,

    144053.318603516,
    95457.798828125,
    92831.6398925781,
    109800.199462891,
    120304.599609375,
    123298.638671875,
    92392.87890625,
    161166.87890625,
    140966.478637695,
    117270.19921875,

    124946.51953125
]



if __name__ == "__main__":

    args = args()

    device = torch.device(args.device)

    N_JOBS = int(args.N_JOBS)
    batch_size = int(args.BATCH)
    init_T = float(args.init_T)
    n_steps=int(args.N_STEPS)

    model = Model(input_node_dim=4, hidden_node_dim=64, input_edge_dim=3, hidden_edge_dim=16)
    model = model.to(device)
    #model.load_state_dict(torch.load("/home/cb/PycharmProjects/masterarbeit_cpu/src/vsp/model/final_model_operational.model"))
    model.load_state_dict(torch.load("/Users/christianburdea/Documents/Studium/Master/Masterarbeit/vsp_rl_impl/src/vsp/model/vsp_reSteps100_rm10_model_final.model", map_location=device))

    inputs = input_reader.load_vsp_envs_from_pickle("vsp_data_100/pickle_test_data/")

    log = [["plan_nr", "Random Cost", "Before Cost", "After Cost", "OG_Random", "OG_RL"]]

    count_beaten_baseline = 0

    for index, vsp_instance in enumerate(inputs):
        print('-----------------------------------------------------------------------------------------------------')
        print('Plan: {} - n_steps: {}'.format(index, n_steps))
        envs = create_batch_env(1, N_JOBS, vsp_instance)

        optimal_cost = exact_solution[index]

        print("optimal cost:", optimal_cost)

        states = envs.reset()
        before_mean_cost = np.mean([env.cost for env in envs.envs])
        print("before mean cost: {} - optimality gap: {}".format(before_mean_cost, before_mean_cost/optimal_cost))

        states, mean_cost = random_init(envs, n_steps, 1, N_JOBS)
        random_cost = mean_cost
        print("random mean cost: {} - optimality gap: {}".format(random_cost, random_cost/optimal_cost))

        states = envs.reset()
        states,history, actions_history, values_history = roll_out(model,envs,states,n_steps)

        after_mean_cost = np.mean([env.cost for env in envs.envs])
        print ("after mean cost: {} - optimality gap: {}".format(after_mean_cost, after_mean_cost/optimal_cost))
        history = np.array(history)

        log.append([index, before_mean_cost, random_cost, after_mean_cost, random_cost/optimal_cost, after_mean_cost/optimal_cost])

        if random_cost > after_mean_cost:
            count_beaten_baseline += 1
            print("-------- success --------")

    log_path = parentdir + '/' + 'log_eval_steps' + str(n_steps) +'_initT'+ str(init_T) +'.csv'

    print("\n Agent better than baseline in {} cases: ".format(count_beaten_baseline))

    print(tabulate(log))

    mean_OG_random = 0
    mean_OG_RL = 0
    for i in range(1,len(log)):
        mean_OG_random += log[i][4]
        mean_OG_RL += log[i][5]

    mean_OG_random = mean_OG_random/ len(inputs)
    mean_OG_RL = mean_OG_RL / len(inputs)

    print("mean optimality gap ranodm: ", mean_OG_random)
    print("mean optimality gap RL: ", mean_OG_RL)

    log.append([0, 0, 0, 0, 0, 0])
    log.append([n_steps,-1,-1,-1, mean_OG_random, mean_OG_RL])

    print(tabulate(log))


    for l in log:
        with open(log_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(l)
