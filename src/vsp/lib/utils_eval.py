import numpy as np
import json
import random
import torch
from torch_geometric.data import Data,DataLoader
from lib.rms import RunningMeanStd
from arguments import args
import argparse
import vsp_env
args = args()

device = torch.device(args.device)

N_JOBS = int(args.N_JOBS)
batch_size = int(args.BATCH)
LR = float(args.LR)

N_ROLLOUT = int(args.N_ROLLOUT)
ROLLOUT_STEPS = int(args.ROLLOUT_STEPS)
N_STEPS = int(args.N_STEPS)

init_T=float(args.init_T)
final_T=float(args.final_T)
RANDOMIZE = False

MAX_DIST = 600000 # Scale for normalizing
MAX_TIME = 2280 # Scale for normalizing

reward_norm = RunningMeanStd()


def create_env(n_jobs, _input=None, epoch=0, raw=0):
    class Env(object):
        def __init__(self, n_jobs, _input=None, raw=None, epoch=0):
            self.n_jobs = n_jobs
            if _input == None:
                raw = 0
                _input = vsp_envs[random.randint(0, len(vsp_envs) - 1)]
                _input['c2'] = (final_T / init_T) ** (1.0 / ROLLOUT_STEPS)
                _input['temperature'] = init_T
                _input['vehicles'][0]['fee_per_dist'] = 120 / 1000
                _input['vehicles'][0]['fee_per_time'] = 60 / 60
                _input['vehicles'][0]['fixed_cost'] = 1000
                _input['vehicles'][0]['tw']['end'] = 4000

            if RANDOMIZE:
                # shift = random.randint(-5,+5)
                shift = 0
                for i, job in enumerate(_input['jobs']):
                    new_tw = _input["jobs"][i]["tw"]["start"] + shift
                    _input["jobs"][i]["tw"]["start"] = new_tw
                    _input["jobs"][i]["tw"]["end"] = new_tw
                    if _input["jobs"][i]["service_time"] < 0:
                        _input["jobs"][i]["service_time"] = 60
                        print("WARNING: Invalid negative service time set to 60 minutes")
                    if new_tw < 250 or new_tw > 1800:
                        new_tw = random.randint(250, 1800)
                        _input["jobs"][i]["tw"]["start"] = new_tw
                        _input["jobs"][i]["tw"]["end"] = new_tw

            self.input = _input
            self.raw = raw
            dist_time = _input['dist_time']

            self.dists = np.array([[[x['dist'] / MAX_DIST] for x in row] for row in dist_time])
            self.times = np.array([[[x['time'] / MAX_TIME] for x in row] for row in dist_time])

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

            nodes = np.zeros((self.n_jobs + 1, 4))
            edges = np.zeros((self.n_jobs + 1, self.n_jobs + 1, 1))

            mapping = {}

            # Embedding
            time_until_job = 0

            for i, (tour, tour_state) in enumerate(zip(tours, states)):
                # print("x x x x x x x x x x x x x x x x x x x x x x x x x")
                service_trips_tour = list(zip(tour, tour_state[1:]))
                for j, (index, s) in enumerate(service_trips_tour):
                    job = jobs[index]
                    solution_job = service_trips_tour[j]
                    job_before = None
                    solution_job_before = None
                    job_after = None
                    solution_job_after = None

                    service_start_time = job['tw']['start']
                    service_end_time = job['tw']['start'] + job['service_time']

                    dist_until_job = s['dist'],
                    dist_until_job = dist_until_job[0]

                    if j == 0:
                        # job_after = jobs[service_trips_tour[j + 1][0]]
                        # solution_job_after = service_trips_tour[j + 1]
                        time_until_job += job["tw"]["start"] - solution_job[1]["wait_time"]

                    elif j == len(service_trips_tour) - 1:
                        job_before = jobs[service_trips_tour[j - 1][0]]
                        # solution_job_before = service_trips_tour[j - 1]
                        time_until_job += job["tw"]["start"] - job_before["tw"]["start"] + job_before["service_time"]

                    else:
                        job_before = jobs[service_trips_tour[j - 1][0]]
                        # job_after = jobs[service_trips_tour[j + 1][0]]
                        # solution_job_before = service_trips_tour[j - 1]
                        # solution_job_after = service_trips_tour[j + 1]
                        time_until_job += job["tw"]["start"] - job_before["tw"]["start"] + job_before["service_time"]

                    loc = job['loc']

                    embedding_information = [
                        service_start_time / MAX_TIME,
                        service_end_time / MAX_TIME,
                        time_until_job / MAX_TIME,
                        dist_until_job / MAX_DIST,
                    ]

                    if embedding_information[0] > 1 or embedding_information[3] > 1:
                        print("WARNING: Normalizing durin embedding not correct\n", embedding_information)

                    '''
                    print("before: ",  solution_job_before)
                    print("current: ", solution_job)
                    print("after: ", solution_job_after)
                    print()
                    print("before: ", job_before)
                    print("current: ", job)
                    print("after: ", job_after)
                    print()

                    print(tabulate([[
                                     "service_start_time",
                                     "service_end_time",
                                     "time_until_job",
                                     "dist_until_job",
                                    ],embedding_information]))
                    print(dist_until_job)
                    print("\n")
                    '''
                    nodes[loc, :] = embedding_information

                    mapping[loc] = (i, j)

            for tour in tours:
                edges[0][tour[0] + 1][0] = 1
                for l, r in zip(tour[0:-1], tour[1:]):
                    edges[l + 1][r + 1][0] = 1
                edges[tour[-1] + 1][0][0] = 1

            # print(len(self.dists))
            # print(len(edges))
            edges = np.stack([self.dists, self.times, edges], axis=-1)
            edges = edges.reshape(-1, 3)

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
            # print("Step done")
            return nodes, edges, reward

    env = Env(n_jobs, _input, epoch=epoch)
    return env


def create_batch_env(batch_size=batch_size,n_jobs=N_JOBS, instance=None):

    class BatchEnv(object):
        def __init__(self,batch_size, instance):
            one_instance = create_env(n_jobs, _input=instance[0], raw=instance[1])
            self.envs = [one_instance for i in range(batch_size)]

        def reset(self):
            rets = [ env.reset() for env in self.envs ]
            return list(zip(*rets))

        def step(self,actions):
            actions = actions.tolist()
            assert(len(actions) == len(self.envs))
            rets = [env.step(act) for env,act in zip(self.envs,actions)]
            return list(zip(*rets))

        def sisr_step(self):
            rets = [env.sisr_step() for env in self.envs]
            return list(zip(*rets))

    return BatchEnv(batch_size, instance)

def create_replay_buffer(n_jobs=99):

    class Buffer(object):
        def __init__(self,n_jobs=n_jobs):
            super(Buffer,self).__init__()
            self.buf_nodes = []
            self.buf_edges = []
            self.buf_actions = []
            self.buf_rewards = []
            self.buf_values = []
            self.buf_log_probs = []
            self.n_jobs = n_jobs

            edges = []
            for i in range(n_jobs+1):
                for j in range(n_jobs+1):
                    edges.append([i,j])

            self.edge_index = torch.LongTensor(edges).T

        def obs(self,nodes,edges,actions,rewards,log_probs,values):
            self.buf_nodes.append(nodes)
            self.buf_edges.append(edges)
            self.buf_actions.append(actions)
            self.buf_rewards.append(rewards)
            self.buf_values.append(values)
            self.buf_log_probs.append(log_probs)

        def compute_values(self,last_v=0,_lambda = 1.0):
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

            return target_vs,advs

        def gen_datas(self,last_v=0,_lambda = 1.0,batch_size=batch_size):
            target_vs,advs = self.compute_values(last_v,_lambda)
            advs = (advs - advs.mean()) / advs.std()
            l,w = target_vs.shape

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
                    data = Data(x=torch.from_numpy(nodes).float(),edge_index=self.edge_index,
                                edge_attr=torch.from_numpy(edges).float(),v=torch.tensor([v]).float(),
                                action=torch.tensor(action).long(),
                                log_prob=torch.tensor([log_prob]).float(),
                                adv = torch.tensor([adv]).float())
                    datas.append(data)

            return datas

        def create_data(self,_nodes,_edges):
            datas = []
            l = len(_nodes)
            for i in range(l):
                nodes = _nodes[i]
                edges = _edges[i]
                data = Data(x=torch.from_numpy(nodes).float(),edge_index=self.edge_index,edge_attr=torch.from_numpy(edges).float())
                datas.append(data)
            dl = DataLoader(datas,batch_size=l)
            return list(dl)[0]

    return Buffer()


def random_init(envs, n_steps, batch_size, n_jobs):

    n_remove_init = 10

    nodes, edges = envs.reset()
    for i in range(n_steps):
        actions = [random.sample(range(0, n_jobs), n_remove_init) for i in range(batch_size)]
        actions = np.array(actions)
        nodes, edges, rewards = envs.step(actions)

    return (nodes, edges), np.mean([env.cost for env in envs.envs])


def roll_out(model,envs,states,n_steps=10,_lambda=0.99,batch_size=batch_size,n_remove=10,is_last=False,greedy=True):
    buffer = create_replay_buffer(N_JOBS)

    with torch.no_grad():
        model.eval()
        nodes,edges = states
        _sum = 0
        _entropy = []

        history = []
        actions_history = []
        values_history = []
        for i in range(n_steps):
            data = buffer.create_data(nodes,edges)
            data = data.to(device)
            actions,log_p,values,entropy = model(data,n_remove,greedy)
            actions_history.append(list(actions))
            values_history.append(envs.envs[0].env.tours())
            new_nodes,new_edges,rewards = envs.step(actions.cpu().numpy())
            rewards = np.array(rewards)
            _sum = _sum + rewards
            rewards = reward_norm(rewards)
            _entropy.append(entropy.mean().cpu().numpy())

#             buffer.obs(nodes,edges,actions.cpu().numpy(),rewards,log_p.cpu().numpy(),values.cpu().numpy())
            nodes,edges = new_nodes,new_edges
            history.append([env.cost for env in envs.envs])

        mean_value = _sum.mean()
#         print ("mean rewards:",mean_value)
#         print ("entropy:",np.mean(_entropy))
#         print ("mean cost:",np.mean([env.cost for env in envs.envs]))

        if not is_last:
#             print ("not last")
            data = buffer.create_data(nodes,edges)
            data = data.to(device)
            actions,log_p,values,entropy = model(data,n_remove,greedy)
            values = values.cpu().numpy()
        else:
            values = 0

#         dl = buffer.gen_datas(values,_lambda = _lambda,batch_size=batch_size)
        #print('nodes', len(nodes))
        #print('edges', len(edges))
        #print('history', len(history))
        #print('actions_history', len(actions_history))
        #print('values_history', len(values_history))
        #return (nodes,edges),history, np.array(actions_history), np.array(values_history)
        return (nodes,edges),history, actions_history, values_history