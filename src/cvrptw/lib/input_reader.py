import pandas as pd
import numpy as np
import random
from tabulate import tabulate
from datetime import datetime
import pickle
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 10000)

from arguments import args
args = args()
import os

N_STEPS = int(args.N_STEPS)

def create_timetable_from_file(path):
    print('Reading timetable...')
    with open(path) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]

    content_seperated = []
    seperated_block = []

    reading_block = False

    #seperates content blocks of files
    for r in content:
        if r[0] == "$":
            reading_block = True
        if r[0] == "*":
            reading_block = False
            if len(r) >1:
                if r[1] == '$':
                    reading_block = True


        if reading_block:
            seperated_block.append(r)
        else:
            if seperated_block != []:
                content_seperated.append(seperated_block)
                seperated_block = []
        if r == content[-1] and reading_block:
            content_seperated.append(seperated_block)

    #seperates list elements by ; to create elementar instances
    for i in range(0, len(content_seperated)):
        for j in range (0, len(content_seperated[i])):
            content_seperated[i][j] = content_seperated[i][j].split(';')

    #for elem in content_seperated:
    #     print(tabulate(elem))

    return content_seperated


def read_vehicle_scheduling_from_file(path):
    with open(path) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]

    content_seperated = []
    seperated_block = []

    reading_block = False

    #seperates content blocks of files
    for r in content:
        if r[0] == "$":
            reading_block = True
        if r[0] == "*":
            reading_block = False
            if len(r) >1:
                if r[1] == '$':
                    reading_block = True


        if reading_block:
            seperated_block.append(r)
        else:
            if seperated_block != []:
                content_seperated.append(seperated_block)
                seperated_block = []
        if r == content[-1] and reading_block:
            content_seperated.append(seperated_block)

    #seperates list elements by ; to create elementar instances
    for i in range(0, len(content_seperated)):
        for j in range (0, len(content_seperated[i])):
            content_seperated[i][j] = content_seperated[i][j].split(';')

    for elem in range(0,len(content_seperated)):
        print(tabulate(content_seperated[elem]))

    return content


def convert_timetable_to_df(timetable):
    #print(timetable)
    timetable = timetable[6]
    timetable = np.asarray(timetable)
    timetable = np.transpose(timetable)

    dataset = pd.DataFrame({'ID': timetable[0][1:]})

    for i in range(1, len(timetable)):
        dataset[timetable[i][0]] = timetable[i][1:]

    dataset = dataset.set_index('ID')
    dataset.index = dataset.index.astype(int)
    dataset['FromStopID'] = dataset['FromStopID'].astype(int)
    dataset['ToStopID'] = dataset['ToStopID'].astype(int)
    dataset['Distance'] = dataset['Distance'].astype(int)
    dataset['DepTime'] = transform_datetime_to_minutes(dataset['DepTime'])
    dataset['DepTime'] = dataset['DepTime'].astype(int)
    dataset['ArrTime'] = transform_datetime_to_minutes(dataset['ArrTime'])
    dataset['ArrTime'] = dataset['ArrTime'].astype(int)

    return dataset


def convert_connections_to_df(timetable):
    timetable = timetable[7]
    timetable = np.asarray(timetable)
    timetable = np.transpose(timetable)

    dataset = pd.DataFrame({'FromStopID': timetable[0][1:]})

    for i in range(1, len(timetable)):
        dataset[timetable[i][0]] = timetable[i][1:]

    dataset['FromStopID'] = dataset['FromStopID'].astype(int)
    dataset['ToStopID'] = dataset['ToStopID'].astype(int)
    dataset['Distance'] = dataset['Distance'].astype(int)
    dataset['RunTime'] = dataset['RunTime'].astype(int)

    return dataset


def transform_datetime_to_minutes(time_string):

    minutes = []

    for str in time_string.tolist():
        datetime_object = datetime.strptime(str[4:], '%H:%M:%S')
        time = datetime_object.time()
        time_in_seconds = time.hour * 60 + time.minute
        minutes.append(time_in_seconds)

    return minutes


def get_dist_time(i_id, j_id, connections):
    dist = 0
    time = 0

    if i_id == j_id:
        dist = 0
        time = 0
        return 0,0
    else:
        connection_from = connections.loc[connections['FromStopID'] == i_id]
        connection_comp = connection_from.loc[connection_from['ToStopID'] == j_id]
        # print(connection)
        try:
            dist = connection_comp.loc[:, 'Distance'].item()
            time = connection_comp.loc[:, 'RunTime'].item()
            #print(i_id, j_id, dist, time / 60)
            # print('---')
            return dist, time/60
        except:
            #print(i_id)
            #print(j_id)
            #print(connection_from)
            #print(connection_comp)
            return False, False


def create_vsp_env_from_file(path):

    timetable = create_timetable_from_file(path)


    depot_id = int(timetable[5][1][1])
    service_trips = convert_timetable_to_df(timetable)
    #service_trips = service_trips[:50]
    #print(service_trips)
    connections = convert_connections_to_df(timetable)
    #print(connections.head())
    service_trips['service_id'] = service_trips.index.values

    # Initialize jobs
    # TODO: implement line. Check whether name or loc can be used for this
    # TODO: Check, what impact the time windows have on the calculation. Is it possible to just use the edges?
    jobs = []
    loc_counter = 1
    for index, row in service_trips.iterrows():
        jobs.append({
        "id": index,
        "loc": loc_counter,
        "name": str(index),
        "x": 1,
        "y": 1,
        "weight": 0,
        "tw": {
            "start": row['DepTime'],
            "end": row['DepTime'],
        },
        "service_time": row['ArrTime'] - row['DepTime'],
        "job_type": "Pickup",
        })
        loc_counter += 1

    # Add depot
    service_trips_depot = service_trips[1:2].copy()
    service_trips_depot['ID'] = 0
    service_trips_depot = service_trips_depot.set_index('ID')
    service_trips_depot['FromStopID'] = depot_id
    service_trips_depot['ToStopID'] = depot_id
    service_trips_depot['x'] = 0
    service_trips_depot['y'] = 0
    service_trips_depot = service_trips_depot.append(service_trips)

    # Initialize dist_time
    # TODO: Check, if it is possible to leave out edges
    amount_nodes = len(jobs)
    dist_time = []
    for index_i, row_i in service_trips_depot.iterrows():
        service_i_dist_time = []
        for index_j, row_j in service_trips_depot.iterrows():
            start = row_i['ToStopID']
            end = row_j['FromStopID']
            dist, time = get_dist_time(start, end, connections)
            service_i_dist_time.append({
                'dist': dist,
                'time': time
            })
        dist_time.append(service_i_dist_time)
        print('> Initialized {}% of nodes'.format(int(round((len(dist_time)/amount_nodes)*100,0))))

    # Initialize adjs
    # TODO: Figure out what the purpose of adjs is
    adjs = []

    for i, job in enumerate(jobs):
        #l = [(_job['id'], dist_time[job['loc']][_job['loc']]['dist']) for j, _job in enumerate(jobs)]
        #print(l)
        l = [(j, dist_time[job['loc']][_job['loc']]['dist']) for j, _job in enumerate(jobs)]
       # print(l)
        l = sorted(l, key=lambda x: x[1])
        l = [x[0] for x in l]
        adjs.append(l)


    init_T = float(args.init_T)
    final_T = float(args.final_T)
    alpha_T = (final_T / init_T) ** (1.0 / N_STEPS)

    v = {
        "cap": 1,
        "tw": {
            "start": 0,
            "end": 1440,
        },
        "start_loc": 0,
        "end_loc": 0,
        "fee_per_dist": 120,
        "fee_per_time": 30,
        "fixed_cost": 0,
        "handling_cost_per_weight": 0.0,
        "max_stops": 0,
        "max_dist": 0,
    }

    input_data = {
        "vehicles": [v],
        "dist_time": dist_time,
        "cost_per_absent": 99999999999999,
        "jobs": jobs,
        "depot": [0,0],
        #"l_max": 100,
        #"c1": 100,
        "adjs": adjs,
        "temperature": 100,
        "c2": alpha_T,
        "sa": True, #Simulated Annealing
    }

    #print(input_data)
    return input_data, 0


def read_optimal_solution(path):
    print('Reading solved vehicle scheduling...')
    with open(path) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]

    content_seperated = []
    seperated_block = []

    reading_block = False

    # seperates content blocks of files
    for r in content:
        if r[0] == "$":
            reading_block = True
        if r[0] == "*":
            reading_block = False
            if len(r) > 1:
                if r[1] == '$':
                    reading_block = True

        if reading_block:
            seperated_block.append(r)
        else:
            if seperated_block != []:
                content_seperated.append(seperated_block)
                seperated_block = []
        if r == content[-1] and reading_block:
            content_seperated.append(seperated_block)

    # seperates list elements by ; to create elementar instances
    for i in range(0, len(content_seperated)):
        for j in range(0, len(content_seperated[i])):
            content_seperated[i][j] = content_seperated[i][j].split(';')

    blocks = content_seperated[0]
    blockelements = content_seperated[1]

    return blocks, blockelements


def calculate_costs_from_solution(blocks, blockelements, connections, vehicle_cost, km_cost):

    vehicle_costs = (len(blocks)-1) * vehicle_cost
    dist_costs = 0
    time_costs = 0 #Implement if needed

    blockelements = np.asarray(blockelements)
    blockelements = np.transpose(blockelements)


    dataset = pd.DataFrame({'BlockID': blockelements[0][1:]})


    for i in range(1, len(blockelements)):
        dataset[blockelements[i][0]] = blockelements[i][1:]

    dataset['BlockID'] = dataset['BlockID'].astype(int)
    dataset['FromStopID'] = dataset['FromStopID'].astype(int)
    dataset['ToStopID'] = dataset['ToStopID'].astype(int)
    dataset['ElementType'] = dataset['ElementType'].astype(int)

    counter = 1

    for index, row in dataset.iterrows():
        if row['ElementType'] != 1 and row['ElementType'] != 9:
            counter +=1
            dep_id = row['FromStopID']
            arr_id = row['ToStopID']
            trip_dist, trip_time = get_dist_time(dep_id, arr_id, connections)
            dist_costs += (trip_dist * km_cost)

    print('#######################################')
    print('Optimal solution')
    print('Total connections: ', counter)
    print('Total blocks: ', (len(blocks)-1))
    print('Total constraint violations: ', 0)
    print('---------------------------------------')
    print('Vehicle cost: ', vehicle_costs)
    print('Kilometer cost: ', dist_costs * km_cost)
    print('Total cost: ', vehicle_costs + dist_costs)
    print('#######################################')

    return vehicle_costs + dist_costs


def get_loc_dict(jobs):
    loc_dict = {}
    for elem in jobs:
        id = elem['id']
        loc = elem['loc']
        loc_dict[loc] = int(id)

    return loc_dict


def check_solution_consistency(tour_loc, jobs, connections, timetable, depot_id, vehicle_cost, km_cost):

    loc_dict = get_loc_dict(jobs)
    dist_cost = 0
    constraint_violations = 0

    timetable = convert_timetable_to_df(timetable)
    for tour in tour_loc:
        last_station = depot_id
        for index, job in enumerate(tour):
            trip_id = loc_dict[job+1]
            service_start_id = timetable.loc[trip_id, 'FromStopID']
            dist, travel_time = get_dist_time(last_station, service_start_id, connections)
            dist_cost += dist
            service_start_time = timetable.loc[trip_id, 'DepTime']
            if last_station != depot_id:
                last_arr_time = timetable.loc[loc_dict[tour[index-1]+1], 'ArrTime']
                if last_arr_time + travel_time > service_start_time:
                    constraint_violations += 1
                    print('------------------------\npos:{}, trip_id: {}'.format(job, trip_id))
                    print('Violation for service {} to {}'.format(last_station, service_start_id))
                    print('last_arr_time: ', last_arr_time)
                    print('travel_time: ', travel_time)
                    print('arr_time: {} > departure_time: {}'.format((last_arr_time + travel_time), service_start_time))
            last_station = timetable.loc[trip_id, 'ToStopID']
        last_dist, last_time = get_dist_time(last_station, depot_id, connections)
        dist_cost += last_dist

    print('#######################################')
    print('Consistency check:')
    print('Total jobs: ', len([item for sublist in tour_loc for item in sublist]))
    print('Total blocks: ', len(tour_loc))
    print('Total constraint violations: ', constraint_violations)
    print('---------------------------------------')
    print('Vehicle cost: ', len(tour_loc) * vehicle_cost)
    print('Kilometer cost: ', dist_cost * km_cost)
    print('Total cost: ', (len(tour_loc) * vehicle_cost)+(dist_cost * km_cost))
    print('#######################################')


def save_plans_as_pickle():

    all_paths = [x for x in os.listdir("vsp_data") if x[:8] == "Fahrplan"and x[-4:] == ".txt"]

    for i, path in enumerate(all_paths):
        plan = create_vsp_env_from_file('vsp_data/' + path)
        with open("vsp_data/pickle_train_data/vsp_plan_nr" + str(i) + '.pkl', 'wb') as output:
            pickle.dump(plan, output, pickle.HIGHEST_PROTOCOL)


def load_vsp_envs_from_pickle(path):
    envs = []
    all_paths = [x for x in os.listdir(path) if x[-4:] == ".pkl"]

    for object_path in all_paths:
        with open(path + "/" +object_path, 'rb') as handle:
            envs.append(pickle.load(handle))
    return envs


if __name__ == "__main__":
    blocks, blockelements = read_optimal_solution("../vsp_data/Umlaufplan_213_1_1_L.txt")

    timetable = create_timetable_from_file("../vsp_data/Fahrplan_213_1_1_L.txt")
    for elem in timetable:
        print(tabulate(elem))
    connections = convert_connections_to_df(timetable)

    #print(tabulate(blocks))
    #print(tabulate(blockelements))

    print(calculate_costs_from_solution(blocks, blockelements, connections, 200000, 1))