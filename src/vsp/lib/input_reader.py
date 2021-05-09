import random

import pandas as pd
import numpy as np
import os, sys
from tabulate import tabulate
from datetime import datetime
import pickle
import random
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 10000)
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from arguments import args
args = args()


N_STEPS = int(args.N_STEPS)
N_JOBS = int(args.N_JOBS)

randomize = True

def create_timetable_from_file(path):
    print('Reading timetables...')
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
    #print(timetables)
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
    dataset['DepTime'] = transform_datetime_to_minutes(dataset['DepTime'])
    dataset['DepTime'] = dataset['DepTime'].astype(int)
    dataset['ArrTime'] = transform_datetime_to_minutes(dataset['ArrTime'])
    dataset['ArrTime'] = dataset['ArrTime'].astype(int)

    if 'Distance' in dataset:
        dataset['Distance'] = dataset['Distance'].astype(int)

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
    dataset['RunTime'] = dataset['RunTime'].astype(int)
    if 'Distance' in dataset:
        dataset['Distance'] = dataset['Distance'].astype(int)

    return dataset


def transform_datetime_to_minutes(time_string):

    minutes = []

    for time_str in time_string.tolist():

        try:
            datetime_object = datetime.strptime(time_str[4:], '%H:%M:%S')
        except:
            datetime_object = datetime.strptime(time_str[2:], '%H:%M:%S')

        day_shift = time_str.split(':')
        day_shift = int(day_shift[0])
        time = datetime_object.time()
        time_in_minutes = (day_shift * 1440) + (time.hour * 60) + time.minute
        minutes.append(time_in_minutes)

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


def create_vsp_env_from_file(path, start_depot=-1):

    timetable = create_timetable_from_file(path)
    #for elem in timetable:
    #    print(tabulate(elem))

    if start_depot == -1:
        depot_id = int(timetable[5][1][1])
        print("Depot ID: ", depot_id)
    else:
        depot_id = start_depot

    vehicle_cost = int(timetable[2][1][3])
    km_cost = int(timetable[2][1][4])
    hour_cost = int(timetable[2][1][5])

    service_trips = convert_timetable_to_df(timetable)

    if len(service_trips.index)>N_JOBS:
        print("Downsizing input problem to N_STEPS: ", N_JOBS)
        service_trips = service_trips.sample(n = 100)
        connections = convert_connections_to_df(timetable)

    if randomize:
        deviation = random.uniform(0.7, 1.3)
        connections["Distance"] = connections["Distance"].apply(lambda x: int(x * deviation))
        connections["RunTime"] = connections["RunTime"].apply(lambda x: int(x * deviation))
        km_cost = random.randint(80, 200)
        hour_cost = random.randint(35, 120)

    service_trips['service_id'] = service_trips.index.values

    # Initialize jobs
    jobs = []
    loc_counter = 1
    for index, row in service_trips.iterrows():
        jobs.append({
        "id": index,
        "loc": loc_counter,
        "name": "ID:"+str(index),
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
        #print('> Initialized {}% of nodes'.format(int(round((len(dist_time)/amount_nodes)*100,0))))

    # Initialize adjs
    # TODO: Figure out what the purpose of adjs is
    adjs = []

    for i, job in enumerate(jobs):
        l = [(j, dist_time[job['loc']][_job['loc']]['dist']) for j, _job in enumerate(jobs)]
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
            "end": 2880,
        },
        "start_loc": 0,
        "end_loc": 0,
        "fee_per_dist": km_cost / 1000,
        "fee_per_time": hour_cost / 60,
        "fixed_cost": vehicle_cost,
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
        "l_max": 0,
        "c1": 0,
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


def calculate_costs_from_solution(blocks, blockelements, timetable):

    connections = convert_connections_to_df(timetable)

    km_cost = int(timetable[2][1][4])
    km_cost = 120
    meter_cost = km_cost / 1000
    hour_cost = int(timetable[2][1][5])
    print("hour_cost: ", hour_cost)
    minute_cost = hour_cost / 60

    dist_costs_total = 0
    time_costs_total = 0

    blockelements = np.asarray(blockelements)
    blockelements = np.transpose(blockelements)


    dataset = pd.DataFrame({'BlockID': blockelements[0][1:]})


    for i in range(1, len(blockelements)):
        dataset[blockelements[i][0]] = blockelements[i][1:]

    dataset['BlockID'] = dataset['BlockID'].astype(int)
    dataset['FromStopID'] = dataset['FromStopID'].astype(int)
    dataset['ToStopID'] = dataset['ToStopID'].astype(int)
    dataset['ElementType'] = dataset['ElementType'].astype(int)

    dataset['DepTime'] = transform_datetime_to_minutes(dataset['DepTime'])
    dataset['DepTime'] = dataset['DepTime'].astype(int)

    dataset['ArrTime'] = transform_datetime_to_minutes(dataset['ArrTime'])
    dataset['ArrTime'] = dataset['ArrTime'].astype(int)

    counter = 1

    #calculate dist related costs
    for index, row in dataset.iterrows():
        if row['ElementType'] != 1 and row['ElementType'] != 9:
            counter +=1
            dep_id = row['FromStopID']
            arr_id = row['ToStopID']
            trip_dist, trip_time = get_dist_time(dep_id, arr_id, connections)
            dist_costs_total += (trip_dist * meter_cost)

    #calculate time related costs
    for index, row in dataset.iterrows():
        if row['ElementType'] != 1:
            time_costs_total += (row['ArrTime'] - row['DepTime']) * minute_cost
        else:
            arr_time = row['ArrTime']
            dep_time = dataset.loc[index+1, :]['DepTime']
            time_costs_total += (dep_time - arr_time) * minute_cost


    print('#######################################')
    print('Optimal solution')
    print('Total connections: ', counter)
    print('Total blocks: ', (len(blocks)-1))
    print('Total constraint violations: ', 0)
    print('---------------------------------------')
    print('Time cost: ', time_costs_total)
    print('Kilometer cost: ', dist_costs_total)
    print('Total cost: ', dist_costs_total + time_costs_total)
    print('#######################################')

    return dist_costs_total + time_costs_total


def get_loc_dict(jobs):
    loc_dict = {}
    for elem in jobs:
        id = elem['id']
        loc = elem['loc']
        loc_dict[loc] = int(id)

    return loc_dict


def check_solution_consistency(tour_loc, jobs, connections, timetable, depot_id, vehicle_cost):

    #print(tabulate(timetable[6]))
    km_cost = int(timetable[2][1][4])
    km_cost = 120
    m_cost = km_cost / 1000

    hour_cost = int(timetable[2][1][5])
    hour_cost = 60
    minute_cost = hour_cost / 60

    loc_dict = get_loc_dict(jobs)
    dist_cost = 0
    constraint_violations = 0

    timetable = convert_timetable_to_df(timetable)

    # Calculate distance costs
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
                    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    print('npos:{}, from trip_id: {} to trip_id: {}'.format(job, tour[index-1], trip_id))
                    print('Violation for service {} to {}'.format(last_station, service_start_id))
                    print('last_arr_time: ', last_arr_time)
                    print('travel_time: ', travel_time)
                    print('arr_time: {} > departure_time: {}'.format((last_arr_time + travel_time), service_start_time))
            last_station = timetable.loc[trip_id, 'ToStopID']
        last_dist, last_time = get_dist_time(last_station, depot_id, connections)
        dist_cost += last_dist

    # Calculate time costs
    total_time = 0
    for tour in tour_loc:
        print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        for index, job in enumerate(tour):
            trip_id = loc_dict[job + 1]
            if index == 0:
                service_start_id = timetable.loc[trip_id, 'FromStopID']
                dist, travel_time = get_dist_time(depot_id, service_start_id, connections)
                total_time += travel_time
                print("anfang:", travel_time)
            elif index == len(tour)-1:
                # Time to last trip
                previous_job = tour[index - 1]
                last_trip_id = loc_dict[previous_job + 1]
                current_trip_id = loc_dict[job + 1]
                last_arr_time = timetable.loc[last_trip_id, 'ArrTime']
                curr_dep_time = timetable.loc[current_trip_id, 'DepTime']
                gap_time = curr_dep_time - last_arr_time
                print("mitte:", gap_time)
                total_time += gap_time

                # Time from last trip to depot
                service_stop_id = timetable.loc[trip_id, 'ToStopID']
                dist, travel_time = get_dist_time(service_stop_id, depot_id, connections)
                total_time += travel_time
                print("ende:", travel_time)
            else:
                previous_job = tour[index-1]
                last_trip_id = loc_dict[previous_job + 1]
                current_trip_id = loc_dict[job + 1]
                last_arr_time = timetable.loc[last_trip_id, 'ArrTime']
                curr_dep_time = timetable.loc[current_trip_id, 'DepTime']
                gap_time = curr_dep_time - last_arr_time
                print("mitte:", gap_time)
                total_time += gap_time

    print('#######################################')
    print('Consistency check:')
    print('Total jobs: ', len([item for sublist in tour_loc for item in sublist]))
    print('Total blocks: ', len(tour_loc))
    print('Total constraint violations: ', constraint_violations)
    print('---------------------------------------')
    print('Vehicle cost: ', len(tour_loc) * vehicle_cost)
    print('Time cost: ', total_time * minute_cost)
    print('Kilometer cost: ', dist_cost * m_cost)
    print('Total cost: ', (len(tour_loc) * vehicle_cost) + (dist_cost * m_cost) + (total_time * minute_cost))
    print('#######################################')


def load_vsp_envs_from_pickle(path):
    envs = []
    all_paths = [x for x in os.listdir(parentdir + '/' +path) if x[-4:] == ".pkl"]

    for object_path in all_paths:
        with open(parentdir + '/' + path + "/" +object_path, 'rb') as handle:
            envs.append(pickle.load(handle))

    return envs


def save_plan_as_pickle(path):
        plan = create_vsp_env_from_file(path)
        with open(parentdir + '/' + "vsp_data_100/dummy_envs/test_vsp_instance.pkl", 'wb') as output:
            pickle.dump(plan, output, pickle.HIGHEST_PROTOCOL)


def multiply_instances_to_pickle( amount_nodes = "100", depot_numbers = 6, multiply_factor = 7):
    all_paths = [x for x in os.listdir(parentdir + '/' + "vsp_data_100/timetables/" + amount_nodes) if x[-4:] == ".txt"]

    counter = 0

    for i, path in enumerate(all_paths):
        print("instance: ", i)
        for depot_nr in range(0,depot_numbers):
            for dep_instance in range(0,multiply_factor + 1):
                print(i,depot_nr, dep_instance)
                plan = create_vsp_env_from_file(parentdir + '/' + "vsp_data_100/timetables/" + amount_nodes + '/' + path, depot_nr)
                with open(parentdir + '/' + "vsp_data_100/pickle_train_data/org"+ amount_nodes + "_vsp_instance_nr" + str(counter) +'_depot'+ str(depot_nr) +'.pkl', 'wb') as output:
                    pickle.dump(plan, output, pickle.HIGHEST_PROTOCOL)
                counter += 1
                print("environments created: ", counter)


if __name__ == "__main__":
    #save_plan_as_pickle("/home/cb/PycharmProjects/masterarbeit_cpu/src/vsp/vsp_data_100/timetables/200/huis_200_2_1_A01.txt")

    from multiprocessing.dummy import Pool as ThreadPool
    pool = ThreadPool(4)

    parameter = ("800", 8, 15, "640", 8, 15, "400", 6, 10,"400", 5, 10)

    #multiply_instances_to_pickle("800", depot_numbers=8, multiply_factor=20)
    #multiply_instances_to_pickle("640", depot_numbers=8, multiply_factor=20)
    #multiply_instances_to_pickle("400", depot_numbers=6, multiply_factor=10)
    #multiply_instances_to_pickle("320", depot_numbers=5, multiply_factor=10)

    #results = pool.map(multiply_instances_to_pickle, parameter)

    #timetable = create_timetable_from_file("/home/cb/PycharmProjects/masterarbeit_cpu/src/vsp/vsp_data_100/timetables/huis_100_2_1_A01.txt")
    #for elem in timetable:
    #    print(tabulate(elem))
    #create_vsp_env_from_file("/home/cb/PycharmProjects/masterarbeit_cpu/src/vsp/vsp_data_100/timetables/huis_100_2_1_A01.txt")


    '''
    blocks, blockelements = read_optimal_solution("/home/cb/PycharmProjects/masterarbeit_cpu/src/vsp/vsp_data_100/solved_schedules/huis_100_2_1_A01_e_M.txt")
    connections = convert_connections_to_df(timetable)

    #print(tabulate(blocks))
    #print(tabulate(blockelements))

    print(calculate_costs_from_solution(blocks, blockelements, timetable))
    '''