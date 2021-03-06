import numpy as np

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)

import json
import csv
import random
import math
import vsp_env as vrp_env
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data,DataLoader
from lib.rms import RunningMeanStd
from arguments import args
import lib.input_reader as reader
import argparse
from multiprocessing.dummy import Pool as ThreadPool
import gc
import timeit
args = args()

device = torch.device(args.device)
N_JOBS = int(args.N_JOBS)
#CAP = int(args.CAP)
batch_size = int(args.BATCH)
#MAX_COORD = int(args.MAX_COORD)
LR = float(args.LR)
#DEPOT_END = int(args.DEPOT_END)
#SERVICE_TIME = int(args.SERVICE_TIME)
#TW_WIDTH = int(args.TW_WIDTH)

N_ROLLOUT = int(args.N_ROLLOUT)
ROLLOUT_STEPS = int(args.ROLLOUT_STEPS)
N_STEPS = int(args.N_STEPS)

init_T=float(args.init_T)
final_T=float(args.final_T)

reward_norm = RunningMeanStd()


PRE_STEPS = int(args.PRE_STEPS)
N_THREADS = int(args.N_THREADS)

MAX_DIST = 600000 # Scale for normalizing
MAX_TIME = 2280 # Scale for normalizing


n_instances = 1

vsp_envs = []

def initialize_vsp_envs(path):
    global vsp_envs
    loaded_vsp_envs = reader.load_vsp_envs_from_pickle(path)
    for instance in loaded_vsp_envs:
        vsp_envs.append(instance[0])

def create_env(n_jobs=N_JOBS,_input=None):

    class Env(object):
        def __init__(self,n_jobs=N_JOBS,_input=None,raw=None):
            self.n_jobs = n_jobs

            raw = 0

            _input = vsp_envs[random.randint(0, len(vsp_envs) - 1)]

            _input['c2'] = (final_T / init_T) ** (1.0 / ROLLOUT_STEPS)
            _input['temperature'] = init_T
            _input['vehicles'][0]['fee_per_dist'] = 120 / 1000
            _input['vehicles'][0]['fee_per_time'] = 60 / 60
            _input['vehicles'][0]['fixed_cost'] = 1000
            _input['vehicles'][0]['tw']['end'] = 4000
            _input['adjs'] = 0


            self.input = _input
            self.raw = raw
            dist_time = _input['dist_time']
            self.dists = np.array([[ [x['dist']/MAX_DIST] for x in row ] for row in dist_time])
            self.times = np.array([[[x['time'] / MAX_TIME] for x in row] for row in dist_time])

        def reset(self):
            self.env = vrp_env.Env(json.dumps(self.input))
            self.mapping = {}
            self.cost = 0.0
            self.best = None
            return self.get_states()

        def get_states(self):
            states = self.env.states()
            tours = self.env.tours()
            jobs = self.input['jobs']
            depot = self.input['depot']

            nodes = np.zeros((self.n_jobs+1, 4))
            edges = np.zeros((self.n_jobs+1, self.n_jobs+1, 1))
            mapping = {}

            time_until_job = 0

            for i,(tour,tour_state) in enumerate(zip(tours,states)):
                service_trips_tour = list(zip(tour, tour_state[1:]))
                for j,(index,s) in enumerate(service_trips_tour):
                    job = jobs[index]
                    solution_job = service_trips_tour[j]

                   # print(job)
                   # print(service_trips_tour)
                   # print(solution_job)
                   # print(s)

                    service_start_time = job['tw']['start']
                    service_end_time = job['tw']['start'] + job['service_time']
                    dist_until_job = s['dist'],
                    dist_until_job = dist_until_job[0]

                    if j == 0:
                        time_until_job += job["tw"]["start"] - solution_job[1]["wait_time"]

                    elif j == len(service_trips_tour)-1:
                        job_before = jobs[service_trips_tour[j - 1][0]]
                        time_until_job += job["tw"]["start"] - job_before["tw"]["start"] + job_before["service_time"]

                    else:
                        job_before = jobs[service_trips_tour[j - 1][0]]
                        time_until_job += job["tw"]["start"] - job_before["tw"]["start"] + job_before["service_time"]

                    loc = job['loc']
                    embedding_information = [
                        service_start_time / MAX_TIME,
                        service_end_time / MAX_TIME,
                        time_until_job / MAX_TIME,
                        dist_until_job / MAX_DIST,
                    ]

                    '''
                    print(service_start_time, service_end_time, time_until_job, dist_until_job)
                    print("service_start_time: ", service_start_time)
                    print("service_end_time: ", service_end_time)
                    print("time_until_job: ", time_until_job)
                    print("dist_until_job: ", dist_until_job)
                    print("")
                    '''

                    nodes[loc, :] = embedding_information

                    mapping[loc] = (i,j)


            for tour in tours:
                edges[0][tour[0]+1][0] = 1
                for l,r in zip(tour[0:-1],tour[1:]):
                    edges[l+1][r+1][0] = 1
                edges[tour[-1]+1][0][0] = 1

            edges = np.stack([self.dists, self.times, edges],axis=-1)
            edges = edges.reshape(-1,3)

            absents = self.env.absents()
            assert len(absents) == 0,"bad input"

            self.mapping = mapping
            self.cost = self.env.cost()
            if self.best is None or self.cost < self.best:
                self.best = self.cost

            return nodes,edges

        def sisr_step(self):
            prev_cost = self.cost
            self.env.sisr_step()
            nodes,edges = self.get_states()
            reward = prev_cost - self.cost
            return nodes,edges,reward

        def step(self,to_remove):
            prev_cost = self.cost
            self.env.step(to_remove)
            nodes,edges = self.get_states()
            reward = prev_cost - self.cost
            return nodes,edges,reward

    env = Env(n_jobs,_input)
    return env

def create_batch_env(batch_size=batch_size,n_jobs=N_JOBS):

    class BatchEnv(object):
        def __init__(self,batch_size=batch_size):
#             _input = create_instance(n_jobs+1)
            self.envs = [ create_env(n_jobs) for i in range(batch_size) ]

        def reset(self):
            rets = [ env.reset() for env in self.envs ]
            return list(zip(*rets))

        '''
        def step(self, actions):
            actions = actions.tolist()
            assert (len(actions) == len(self.envs))
            # rets = [env.step(act) for env, act in zip(self.envs, actions)]
            list_env_act = [[env, act] for env, act in zip(self.envs, actions)]
            
            def step_parallel(arr):
                return arr[0].step(arr[1])

            pool = ThreadPool(N_THREADS)
            rets = pool.map(step_parallel, list_env_act)
            pool.close()
            return list(zip(*rets))
        '''

        def step(self, actions):
            actions = actions.tolist()
            assert (len(actions) == len(self.envs))
            rets = [env.step(act) for env, act in zip(self.envs, actions)]
            return list(zip(*rets))

        def sisr_step(self):
            rets = [env.sisr_step() for env in self.envs]
            return list(zip(*rets))

    return BatchEnv(batch_size)

def create_replay_buffer(n_jobs=N_JOBS):

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

def roll_out(model,envs,states,n_steps=10,_lambda=0.99,batch_size=batch_size,n_remove=10,is_last=False,greedy=False):
    buffer = create_replay_buffer()
    with torch.no_grad():
        model.eval()
        nodes,edges = states
        _sum = 0
        _entropy = []

        for i in range(n_steps):
            data = buffer.create_data(nodes,edges)
            data = data.to(device)
            actions,log_p,values,entropy = model(data,n_remove,greedy,n_instances)
#             print (actions)
            new_nodes,new_edges,rewards = envs.step(actions.cpu().numpy())
            rewards = np.array(rewards)
            _sum = _sum + rewards
            rewards = reward_norm(rewards)
            _entropy.append(entropy.mean().cpu().numpy())

            buffer.obs(nodes,edges,actions.cpu().numpy(),rewards,log_p.cpu().numpy(),values.cpu().numpy())
            nodes,edges = new_nodes,new_edges

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

        dl = buffer.gen_datas(values,_lambda = _lambda,batch_size=batch_size)
        return dl,(nodes,edges)

def train_once(model,opt,dl,epoch,step,alpha=1.0):
    model.train()

    losses = []
    loss_vs = []
    loss_ps = []
    _entropy = []

    for i,batch in enumerate(dl):
        batch = batch.to(device)
        batch_size = batch.num_graphs
#         print (batch.action.shape)
        actions = batch.action.reshape((batch_size,-1))
        log_p,v,entropy = model.evaluate(batch,actions)
        _entropy.append(entropy.mean().item())

        target_vs = batch.v.squeeze(-1)
        old_log_p = batch.log_prob.squeeze(-1)
        adv =  batch.adv.squeeze(-1)

        loss_v = ((v - target_vs) ** 2).mean()

        ratio = torch.exp(log_p-old_log_p)
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

    print ("epoch:",epoch,"step:",step,"loss_v:",np.mean(loss_vs),"loss_p:",np.mean(loss_ps),"loss:",np.mean(losses),"entropy:",np.mean(_entropy))

def eval_random(epochs,envs,n_steps=10):

    def eval_once(epoch,n_instance=n_instances,n_steps=n_steps,batch_size=batch_size,alpha=1.0):
        nodes,edges = envs.reset()
        _sum = np.zeros(n_instance)
        for i in range(n_steps):
            actions = [ random.sample(range(0, N_JOBS),10) for i in range(n_instance) ]
            actions = np.array(actions)
            new_nodes,new_edges,rewards = envs.step(actions)
            _sum += rewards

        return np.mean([env.cost for env in envs.envs])

    print ("<<<<<<<<<<===================== random mean cost:",np.mean([eval_once(i) for i in range(epochs)]))

def random_init(envs,n_steps,n_instance=n_instances):
    nodes,edges = envs.reset()
    for i in range(n_steps):
        actions = [ random.sample(range(0, N_JOBS),10) for i in range(n_instance) ]
        actions = np.array(actions)
        nodes,edges,rewards = envs.step(actions)

    return (nodes,edges),np.mean([env.cost for env in envs.envs])

def train(model,epochs,n_rollout,rollout_steps,train_steps,n_remove=10):
    initialize_vsp_envs('vsp_data_100/pickle_train_data/data_collection_1')
    #initialize_vsp_envs('vsp_data_100/pickle_train_data/data_collection_2')
    #initialize_vsp_envs('vsp_data_100/pickle_train_data/data_collection_3')
    #initialize_vsp_envs('vsp_data_100/dummy_envs')

    opt = torch.optim.Adam(model.parameters(),LR)

    start = timeit.default_timer()
    log = [["Epoch", "Before Cost", "After Cost"]]

    for epoch in range(epochs):
        start = timeit.default_timer()

        envs = create_batch_env(n_instances,N_JOBS)
        states,mean_cost = random_init(envs,PRE_STEPS)
        before_mean_cost = np.mean([env.cost for env in envs.envs])
        print('\n-----------------------------------------------------------------------------------------------------')
        print("> before mean cost " + ": " + str(before_mean_cost))


        all_datas = []
        for i in range(n_rollout):
            is_last = (i == n_rollout-1)
            datas,states = roll_out(model,envs,states,rollout_steps,is_last=False)
            all_datas.extend(datas)


        '''
        def rollout_parallel(r):
            ret_data = []
            gc.collect()
            datas, _ = roll_out(model=model, envs=envs, states=states, n_steps=rollout_steps,
                                n_remove=n_remove, is_last=False)
            ret_data.extend(datas)
            return ret_data[0]

        all_datas = []
        pool = ThreadPool(2)
        all_datas.extend(pool.map(rollout_parallel, list(range(0, n_rollout + 1))))
        pool.close()
        '''



        dl = DataLoader(all_datas,batch_size=batch_size,shuffle=True)
        for j in range(train_steps):
            gc.collect()
            train_once(model,opt,dl,epoch,0)


        if epoch % 10 == 0:
            eval_random(3,envs,n_rollout*rollout_steps+PRE_STEPS)

        if epoch % 100 == 0:
            torch.save(model.state_dict(),
                       parentdir + '/' + "model/vsp_preSteps{}_sa{}_model_epoch_{}.model".format(str(PRE_STEPS),
                                                                                                 str(init_T),
                                                                                                 epoch))
        cost = np.mean([env.cost for env in envs.envs])
        log.append([[epoch,before_mean_cost, cost]])

        time = (timeit.default_timer() - start) / 60
        print("> improvement: {}% mean cost: {} minutes: {}".format(round((1 - (cost / before_mean_cost)) * 100, 2), cost, time))

    torch.save(model.state_dict(), parentdir + '/' + "model/vsp_preSteps{}_sa{}_model_final.model".format(str(PRE_STEPS),str(init_T)))

    log_path = parentdir + '/log_prod_preSteps{}_sa{}.csv'.format(str(PRE_STEPS), str(init_T))
    for l in log:
        with open(log_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(l)
