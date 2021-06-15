import numpy as np
import json
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import random
import gc
import torch
from torch_geometric.data import Data, DataLoader
from lib.rms import RunningMeanStd
from arguments import args
from tabulate import tabulate
import lib.input_reader as reader
import csv
import timeit
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

args = args()

device = torch.device(args.device)
N_JOBS = int(args.N_JOBS)
BATCH_SIZE = int(args.BATCH)
LR = float(args.LR)
init_T = float(args.init_T)
final_T = float(args.final_T)
N_STEPS = float(args.N_STEPS)
EVAL_MODE = str(args.EVAL_MODE)
RANDOMIZE = str(args.RANDOMIZE)
n_instances = 128
if EVAL_MODE == "operational":
    import vsp_custom_env as vsp_env
else:
    import vrp_env as vsp_env

reward_norm = RunningMeanStd()
vsp_envs = []



def initialize_vsp_envs(path):
    global vsp_envs
    loaded_vsp_envs = reader.load_vsp_envs_from_pickle(path)
    for instance in loaded_vsp_envs:
        vsp_envs.append(instance[0])


def create_env(n_jobs, _input=None, epoch = 0):
    class Env(object):
        def __init__(self, n_jobs, _input=None, raw=None, epoch = 0):
            self.n_jobs = n_jobs
            if _input == None:
                raw = 0
                _input = vsp_envs[random.randint(0, len(vsp_envs) - 1)]
                _input['c2'] = (final_T / init_T) ** (1.0 / N_STEPS)
                _input['temperature'] = init_T
                if EVAL_MODE == "operational":
                    _input['vehicles'][0]['fixed_costs'] = 0
                else:
                    _input['vehicles'][0]['fee_per_time'] = 0
                _input['vehicles'][0]['tw']['end'] = 4000



            if RANDOMIZE:
                #shift = random.randint(-5,+5)
                shift = 0
                for i, job in enumerate(_input['jobs']):
                    new_tw = _input["jobs"][i]["tw"]["start"] + shift
                    _input["jobs"][i]["tw"]["start"] = new_tw
                    _input["jobs"][i]["tw"]["end"] = new_tw
                    if _input["jobs"][i]["service_time"] < 0:
                        _input["jobs"][i]["service_time"] = 60
                        print("WARNING: Invalid negative service time set to 60 minutes")
                    if new_tw < 250 or new_tw > 1800:
                        new_tw = random.randint(250,1800)
                        _input["jobs"][i]["tw"]["start"] = new_tw
                        _input["jobs"][i]["tw"]["end"] = new_tw

                #_input, raw = reader.create_vsp_env_from_file("vsp_data/Fahrplan_213_1_1_L.txt")

            self.input = _input
            self.raw = raw
            dist_time = _input['dist_time']

            # self.dists = np.array([[ [x['dist']/MAX_DIST] for x in row ] for row in dist_time]) Original
            distances_custom = [item['dist'] for sublist in dist_time for item in sublist]
            self.max_dist_custom = max(distances_custom)
            self.dists = np.array([[[x['dist'] / self.max_dist_custom] for x in row] for row in dist_time])

        def reset(self):
            self.env = vsp_env.Env(json.dumps(self.input))
            self.mapping = {}
            self.cost = 0.0
            self.best = None
            return self.get_states()

        def get_states(self):
            # {'id': 1, 'loc': 2, 'name': '1', 'x': 92, 'y': 73, 'weight': 8, 'tw': {'start': 0, 'end': 10000}, 'service_time': 0, 'job_type': 'Pickup'}
            # {'dist': 8.5440034866333, 'time': 8.5440034866333, 'stops': 1, 'time_slack': 9615.22138046875, 'wait_time': 0.0, 'service_time': 0.0, 'loc': 2, 'weight': 8.0}
            states = self.env.states()
            tours = self.env.tours()
            self.vsp_tours = self.env.tours()
            jobs = self.input['jobs']
            self.vsp_jobs = self.input['jobs']
            depot = self.input['depot']

            nodes = np.zeros((self.n_jobs + 1, 8))
            edges = np.zeros((self.n_jobs + 1, self.n_jobs + 1, 1))

            mapping = {}

            for i, (tour, tour_state) in enumerate(zip(tours, states)):
                for j, (index, s) in enumerate(zip(tour, tour_state[1:])):
                    job = jobs[index]
                    loc = job['loc']
                    nodes[loc, :] = [job['weight'] / 1, s['weight'] / 1, s['dist'] / self.max_dist_custom,
                                     s['time'] / self.max_dist_custom, job['tw']['start'] / self.max_dist_custom,
                                     job['tw']['end'] / self.max_dist_custom, s['time'] / self.max_dist_custom,
                                     s['time_slack'] / self.max_dist_custom]
                    mapping[loc] = (i, j)

            for tour in tours:
                edges[0][tour[0] + 1][0] = 1
                for l, r in zip(tour[0:-1], tour[1:]):
                    edges[l + 1][r + 1][0] = 1
                edges[tour[-1] + 1][0][0] = 1

            # print(len(self.dists))
            # print(len(edges))
            edges = np.stack([self.dists, edges], axis=-1)
            edges = edges.reshape(-1, 2)

            absents = self.env.absents()
            if len(absents) != 0:
                print(self.vsp_tours)
                for elem in self.vsp_jobs:
                    print(elem)
                print(tabulate(self.vsp_jobs))
                print(self.vsp_tours)
            assert len(absents) == 0, "bad input"

            self.mapping = mapping
            self.cost = self.env.cost()
            if self.best is None or self.cost < self.best:
                self.best = self.cost


            return nodes, edges

        def sisr_step(self):
            prev_cost = self.cost
            self.env.sisr_step()
            nodes, edges = self.get_states()
            reward = prev_cost - self.cost
            return nodes, edges, reward

        def step(self, to_remove):
            prev_cost = self.cost
            self.env.step(to_remove)
            nodes, edges = self.get_states()
            reward = prev_cost - self.cost
            # print(self.vsp_tours)
            return nodes, edges, reward

    env = Env(n_jobs, _input, epoch = epoch)
    return env


def create_batch_env(batch_size, n_jobs, epoch = 0):
    class BatchEnv(object):
        def __init__(self, batch_size):
            # _input = create_instance(n_jobs+1)
            # self.envs = [create_env(n_jobs, None, test) for i in range(batch_size)]
            e = create_env(n_jobs, None, epoch = epoch)
            envs = []
            for i in range(0, batch_size):
                envs.append(e)
            # print(len(envs))
            self.envs = envs

        def reset(self):
            rets = [env.reset() for env in self.envs]
            return list(zip(*rets))

        def step(self, actions):
            actions = actions.tolist()
            assert (len(actions) == len(self.envs))
            rets = [env.step(act) for env, act in zip(self.envs, actions)]
            return list(zip(*rets))

        def sisr_step(self):
            rets = [env.sisr_step() for env in self.envs]
            return list(zip(*rets))

    return BatchEnv(batch_size)


def create_replay_buffer(n_jobs):
    class Buffer(object):
        def __init__(self, n_jobs=n_jobs):
            super(Buffer, self).__init__()
            self.buf_nodes = []
            self.buf_edges = []
            self.buf_actions = []
            self.buf_rewards = []
            self.buf_values = []
            self.buf_log_probs = []
            self.n_jobs = n_jobs

            edges = []
            for i in range(n_jobs + 1):
                for j in range(n_jobs + 1):
                    edges.append([i, j])

            self.edge_index = torch.LongTensor(edges).T

        def obs(self, nodes, edges, actions, rewards, log_probs, values):
            self.buf_nodes.append(nodes)
            self.buf_edges.append(edges)
            self.buf_actions.append(actions)
            self.buf_rewards.append(rewards)
            self.buf_values.append(values)
            self.buf_log_probs.append(log_probs)

        def compute_values(self, last_v=0, _lambda=1.0):
            rewards = np.array(self.buf_rewards)
            #             rewards = (rewards - rewards.mean()) / rewards.std()
            pred_vs = np.array(self.buf_values)

            target_vs = np.zeros_like(rewards)
            advs = np.zeros_like(rewards)

            #             print (rewards.shape,target_vs.shape,advs.shape,pred_vs.shape)

            v = last_v
            for i in reversed(range(rewards.shape[0])):
                v = rewards[i] + _lambda * v
                target_vs[i] = v
                adv = v - pred_vs[i]
                advs[i] = adv

            return target_vs, advs

        def gen_datas(self, last_v=0, _lambda=1.0):
            target_vs, advs = self.compute_values(last_v, _lambda)
            advs = (advs - advs.mean()) / advs.std()
            l, w = target_vs.shape

            datas = []
            for i in range(l):
                for j in range(w):
                    nodes = self.buf_nodes[i][j]
                    edges = self.buf_edges[i][j]
                    action = self.buf_actions[i][j]
                    v = target_vs[i][j]
                    adv = advs[i][j]
                    log_prob = self.buf_log_probs[i][j]
                    #                     print (nodes.dtype,self.edge_index.dtype,edges.dtype,q,action)
                    data = Data(x=torch.from_numpy(nodes).float(), edge_index=self.edge_index,
                                edge_attr=torch.from_numpy(edges).float(), v=torch.tensor([v]).float(),
                                action=torch.tensor(action).long(),
                                log_prob=torch.tensor([log_prob]).float(),
                                adv=torch.tensor([adv]).float())
                    datas.append(data)

            return datas

        def create_data(self, _nodes, _edges):
            datas = []
            l = len(_nodes)
            for i in range(l):
                nodes = _nodes[i]
                edges = _edges[i]
                data = Data(x=torch.from_numpy(nodes).float(), edge_index=self.edge_index,
                            edge_attr=torch.from_numpy(edges).float())
                datas.append(data)
            dl = DataLoader(datas, batch_size=l)
            return list(dl)[0]

    return Buffer()


def roll_out(model, envs, states, n_jobs, rollout_steps=10, _lambda=0.99, n_remove=10, is_last=False, greedy=False):
    buffer = create_replay_buffer(n_jobs)
    with torch.no_grad():
        model.eval()
        nodes, edges = states
        _sum = 0
        _entropy = []

        for i in range(rollout_steps):
            data = buffer.create_data(nodes, edges)
            data = data.to(device)
            actions, log_p, values, entropy = model(data, n_remove, greedy, 128)
            #             print (actions)
            new_nodes, new_edges, rewards = envs.step(actions.cpu().numpy())
            rewards = np.array(rewards)
            _sum = _sum + rewards
            rewards = reward_norm(rewards)
            _entropy.append(entropy.mean().cpu().numpy())

            buffer.obs(nodes, edges, actions.cpu().numpy(), rewards, log_p.cpu().numpy(), values.cpu().numpy())
            nodes, edges = new_nodes, new_edges

        mean_value = _sum.mean()


        # print("mean rewards:",mean_value)
        # print("entropy:",np.mean(_entropy))
        # print("mean cost:",np.mean([env.cost for env in envs.envs]))
        # print('---')

        if not is_last:
            #             print ("not last")
            data = buffer.create_data(nodes, edges)
            data = data.to(device)
            actions, log_p, values, entropy = model(data, n_remove, greedy)
            values = values.cpu().numpy()
        else:
            values = 0

        dl = buffer.gen_datas(values, _lambda=_lambda)
        return dl, (nodes, edges)


def train_once(model, opt, dl, epoch, step, alpha=1.0):
    model.train()

    losses = []
    loss_vs = []
    loss_ps = []
    _entropy = []

    for i, batch in enumerate(dl):
        batch = batch.to(device)
        batch_size = batch.num_graphs
        # print (batch.action.shape)
        actions = batch.action.reshape((batch_size, -1))
        log_p, v, entropy = model.evaluate(batch, actions)
        _entropy.append(entropy.mean().item())

        target_vs = batch.v.squeeze(-1)
        old_log_p = batch.log_prob.squeeze(-1)
        adv = batch.adv.squeeze(-1)

        loss_v = ((v - target_vs) ** 2).mean()

        ratio = torch.exp(log_p - old_log_p)
        obj = ratio * adv
        obj_clipped = ratio.clamp(1.0 - 0.2,
                                  1.0 + 0.2) * adv
        loss_p = -torch.min(obj, obj_clipped).mean()
        loss = loss_p + alpha * loss_v

        losses.append(loss.item())
        loss_vs.append(loss_v.item())
        loss_ps.append(loss_p.item())

        opt.zero_grad()
        loss.backward()
        opt.step()

    print("epoch:", epoch, "step:", step, "loss_v:", np.mean(loss_vs), "loss_p:", np.mean(loss_ps), "loss:",
          np.mean(losses), "entropy:", np.mean(_entropy))


def eval_random(epochs, envs, n_steps, n_instances, n_jobs):
    def eval_once(epoch):
        nodes, edges = envs.reset()
        _sum = np.zeros(n_instances)
        for i in range(n_steps):
            #print("Achtung: Eval_Random Funktion noch anpassen")
            actions = [random.sample(range(0, n_jobs), 10) for i in range(n_instances)]
            actions = np.array(actions)
            new_nodes, new_edges, rewards = envs.step(actions)
            _sum += rewards

        return np.mean([env.cost for env in envs.envs])

    print(">>random mean cost:", np.mean([eval_once(i, ) for i in range(epochs)]))


def random_init(envs, n_steps, n_jobs):

    n_remove_init = 10

    nodes, edges = envs.reset()
    for i in range(n_steps):
        actions = [random.sample(range(0, n_jobs), n_remove_init) for i in range(n_instances)]
        actions = np.array(actions)
        nodes, edges, rewards = envs.step(actions)

    return (nodes, edges), np.mean([env.cost for env in envs.envs])


def train(model, epochs, n_rollout, rollout_steps, train_steps, n_remove):
    opt = torch.optim.Adam(model.parameters(), LR)
    start = timeit.default_timer()

    #initialize_vsp_envs('vsp_data_100/pickle_train_data/data_collection_1')
    #initialize_vsp_envs('vsp_data_100/pickle_train_data/data_collection_2')
    #initialize_vsp_envs('vsp_data_100/pickle_train_data/data_collection_3')
    initialize_vsp_envs('vsp_data_100/dummy_envs')



    log = [["Epoch", "Before Cost", "After Cost"]]

    print('Seconds for data loading: ', timeit.default_timer() - start)

    for epoch in range(epochs):
        start = timeit.default_timer()
        gc.collect()

        envs = create_batch_env(n_instances, N_JOBS, epoch=epoch)

        pre_steps = 100
        states, mean_cost = random_init(envs, pre_steps, N_JOBS)
        #envs.reset()
        before_mean_cost = np.mean([env.cost for env in envs.envs])

        print('\n-----------------------------------------------------------------------------------------------------')
        print("> before mean cost " + EVAL_MODE + ": " + str(before_mean_cost))
        before_cost = np.mean([env.cost for env in envs.envs])

        all_datas = []
        for i in range(n_rollout):
            datas, states = roll_out(model=model, envs=envs, states=states, rollout_steps=rollout_steps, n_jobs=N_JOBS, n_remove=n_remove, is_last=False)
            all_datas.extend(datas)

        gc.collect()

        dl = DataLoader(all_datas, BATCH_SIZE, shuffle=True)
        for j in range(train_steps):
            gc.collect()
            train_once(model, opt, dl, epoch, 0)

        mean = np.mean([env.cost for env in envs.envs])

        time = (timeit.default_timer() - start) / 60
        print("> improvement: {}% mean cost: {} minutes: {}".format(round((1-(mean/before_mean_cost))*100,2),mean, time))
        cost = np.mean([env.cost for env in envs.envs])
        log.append([[epoch,before_cost, cost]])

        if epoch % 10 == 0:
            eval_random(3, envs, n_rollout * rollout_steps + pre_steps, n_instances, N_JOBS)
        print('-----------------------------------------------------------------------------------------------------')

        if epoch % 100 == 0:
            torch.save(model.state_dict(), parentdir + '/' + "model/vsp_" + EVAL_MODE + "_model_epoch_%s.model" % epoch)

    torch.save(model.state_dict(), parentdir + '/' + "model/vsp_" + EVAL_MODE + "_model_final.model")

    log_path = parentdir + '/' + 'log_prod_' + EVAL_MODE + '.csv'
    for l in log:
        with open(log_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(l)