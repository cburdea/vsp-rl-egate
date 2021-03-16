import pandas as pd
import numpy as np
from tabulate import tabulate
from datetime import datetime
from arguments import args
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 10000)
args = args()


class Timetable:

    stpopoints = None
    lines = None
    vehicle_type = None
    vehicletype_group = None
    vehtype_2_vehtype_group = None
    vehicle_captype_to_stoppoint = None
    service_journey = None
    deadrun_type = None
    reliefpoint = None

    def __init__(self):
        print('tbd')


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

    #for elem in range(0,len(content_seperated)):
    #    print(tabulate(content_seperated[elem]))

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
    else:
        connection = connections.loc[connections['FromStopID'] == i_id]
        connection = connection.loc[connection['ToStopID'] == j_id]
        # print(connection)
        try:
            dist = connection.loc[:, 'Distance'].item()
            time = connection.loc[:, 'RunTime'].item()
           # print(time)
           # print('---')
        except:
            print(i_id)
            print(j_id)
            print(connection)
            print()


    return dist, time


def create_vsp_env_from_file(path, vehicle_cap=1, depot_id = 168):

    timetable = create_timetable_from_file(path)

    service_trips = convert_timetable_to_df(timetable)
    #service_trips = service_trips[:10]
    connections = convert_connections_to_df(timetable)
    service_trips['service_id'] = service_trips.index.values

    # Initialize jobs
    # TODO: implement line. Check whether name or loc can be used for this
    # TODO: Check, what impact the time windows have on the calculation. Is it possible to just use the edges?
    jobs = []
    loc = 1
    for index, row in service_trips.iterrows():
        jobs.append({
        "id": index,
        "loc": loc,
        "name": str(index),
        "x": 0,
        "y": 0,
        "weight": 0,
        "tw": {
            "start": row['DepTime'],
            "end": row['ArrTime'],
        },
        "service_time": row['ArrTime'] - row['DepTime'],
        "job_type": "Pickup",
        })
    loc += 1

    amount_nodes = len(jobs)

    service_trips_depot = service_trips[1:2].copy()
    service_trips_depot['ID'] = 0
    service_trips_depot = service_trips_depot.set_index('ID')
    service_trips_depot['FromStopID'] = depot_id
    service_trips_depot['ToStopID'] = depot_id
    service_trips_depot = service_trips_depot.append(service_trips)

    # Initialize dist_time
    # TODO: Check, if it is possible to leave out edges
    dist_time = []
    for index_i, row_i in service_trips_depot.iterrows():
        service_i_dist_time = []
        for index_j, row_j in service_trips_depot.iterrows():
            start = row_i['ToStopID']
            end = row_j['FromStopID']
            dist, time = get_dist_time(start,end, connections)
            service_i_dist_time.append({
                'dist': dist,
                'time': time/60
            })
        dist_time.append(service_i_dist_time)
        print('> Initialized {}% of nodes'.format(int(round((len(dist_time)/amount_nodes)*100,0))))

    # Initialize adjs
    # TODO: Figure out what the purpose of adjs is
    adjs = []
    for i, job in enumerate(jobs):
        l = [(_job['id'], dist_time[job['loc']][_job['loc']]['dist']) for j, _job in enumerate(jobs)]
        l = sorted(l, key=lambda x: x[1])
        l = [x[0] for x in l]
        adjs.append(l)


    N_STEPS = int(args.N_STEPS)
    init_T = float(args.init_T)
    final_T = float(args.final_T)
    alpha_T = (final_T / init_T) ** (1.0 / N_STEPS)

    v = {
        "cap": vehicle_cap,
        "tw": {
            "start": 0,
            "end": 999999,
        },
        "start_loc": depot_id,
        "end_loc": depot_id,
        "fee_per_dist": 1,
        "fee_per_time": 0,
        "fixed_cost": 0,
        "handling_cost_per_weight": 1.0,
        "max_stops": 0,
        "max_dist": 0,
    }

    input_data = {
        "vehicles": [v],
        "dist_time": dist_time,
        "cost_per_absent": 1000,
        "jobs": jobs,
        "depot": [0,0],
        "l_max": 10,
        "c1": 10,
        "adjs": adjs,
        "temperature": 100,
        "c2": alpha_T,
        "sa": True,
    }


    print(input_data)
    return input_data, 0

#print(create_vsp_env_from_file("../vsp_data/Fahrplan_213_1_1_L.txt"))