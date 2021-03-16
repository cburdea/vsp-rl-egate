import networkx as nx
import pickle
import matplotlib.pyplot as plt
from tabulate import tabulate
import pandas as pd
import numpy as np
import input_reader
from datetime import datetime
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


timetable_df = None
deadhead_df = None


def load_graph(path):
    with open(path, 'rb') as handle:
        obj= pickle.load(handle)
        return obj


def save_graph(path, obj):
    with open(path, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def create_graph_from_timetable_and_connections(timetable, deadheads, depot_nr):
    print('Creating graph:...')
    G = nx.DiGraph()

    # Initialize all service trips as nodes
    for trip in list(timetable.index.values):
        G.add_node(trip, FromStopID=timetable.loc[[trip]]['FromStopID'].item(),
                   ToStopID=timetable.loc[[trip]]['ToStopID'].item(),
                   DepTime=timetable.loc[[trip]]['DepTime'].item(),
                   ArrTime=timetable.loc[[trip]]['ArrTime'].item())

    amount_nodes = len(list(G.nodes))
    visited_nodes = 0

    # add weights by distance
    for i in list(G.nodes):
        visited_nodes += 1
        for j in list(G.nodes):
            if i != j:
                i_stop = G.nodes[i]['ToStopID']
                i_arr_time = G.nodes[i]['ArrTime']
                j_stop = G.nodes[j]['FromStopID']
                j_dep_time = G.nodes[j]['DepTime']
                if G.nodes[i]['ToStopID'] != G.nodes[j]['FromStopID']:
                    connection = deadheads.loc[deadheads['FromStopID'] == i_stop]
                    connection = connection.loc[connection['ToStopID'] == j_stop]
                    distance = connection.loc[:, 'Distance'].item()
                    run_time = connection.loc[:, 'RunTime'].item()
                    if i_arr_time + run_time <= j_dep_time:
                        G.add_edge(i, j, weight=distance)
                else:
                    if i_arr_time <= j_dep_time:
                        G.add_edge(i, j, weight=0)

        if visited_nodes % 5 == 0:
            print('>Initialized {} nodes of {} total nodes'.format(visited_nodes, amount_nodes))

    # Add Depots
    G.add_node(0, FromStopID=depot_nr, ToStopID=depot_nr)
    G.add_node(99999, FromStopID=depot_nr, ToStopID=depot_nr)

    amount_nodes = len(list(G.nodes))

    G.add_edge(99999, 0, weight=-1)

    # Initialize edges from depot to service trips
    visited_nodes = 0
    for j in list(G.nodes):
        visited_nodes += 1
        j_stop = G.nodes[j]['FromStopID']
        if j_stop != depot_nr:
            distance = deadheads.loc[deadheads['FromStopID'] == depot_nr]
            distance = distance.loc[distance['ToStopID'] == j_stop]
            distance = distance.loc[:, 'Distance'].item()
            G.add_edge(0, j, weight=distance)
        print('> Initialized node {} of {} total nodes'.format(visited_nodes, amount_nodes))

    # Initialize edges from service trips to depots
    visited_nodes = 0
    for i in list(G.nodes):
        visited_nodes += 1
        i_stop = G.nodes[i]['ToStopID']
        if i_stop != depot_nr:
            distance = deadheads.loc[deadheads['FromStopID'] == i_stop]
            distance = distance.loc[distance['ToStopID'] == depot_nr]
            distance = distance.loc[:, 'Distance'].item()
            G.add_edge(i, 99999, weight=distance)
        print('> Initialized node {} of {} total nodes'.format(visited_nodes, amount_nodes))

    return G


def test_graph_correctness(g, deadheads, timetable):

    # Check, if all service trips are included
    print(len(timetable.index) + 2)
    print(len(g.nodes))

    print(len(g.edges))

    # Check, if legal edges are included
    try:
        distance = deadheads.loc[deadheads['FromStopID'] == 16]
        distance = distance.loc[distance['ToStopID'] == 88]
        print(distance)
        edge1 = g[8][185]
        print(edge1)
    except:
        print('No Node from 8 -> 185')

    print('')

    try:
        distance = deadheads.loc[deadheads['FromStopID'] == 168]
        distance = distance.loc[distance['ToStopID'] == 5]
        print(distance)
        edge1 = g[0][8]
        print(edge1)
    except:
        print('No Node from 8 -> 8')

    print('')

    try:
        distance = deadheads.loc[deadheads['FromStopID'] == 89]
        distance = distance.loc[distance['ToStopID'] == 89]
        print(distance)
        edge1 = g[185][585]
        print(edge1)
    except:
        print('No Node from 89 -> 89')

    print('')

    try:
        distance = deadheads.loc[deadheads['FromStopID'] == 16]
        distance = distance.loc[distance['ToStopID'] == 4]
        print(distance)
        edge1 = g[81][8]
        print(edge1)
    except:
        print('No Node from 81 -> 8')

    print('')


    try:
        distance = deadheads.loc[deadheads['FromStopID'] == 5]
        distance = distance.loc[distance['ToStopID'] == 5]
        print(distance)
        edge1 = g[81][8]
        print(edge1)
    except:
        print('No Node from 81 -> 8')

    print('')

    try:
        distance = deadheads.loc[deadheads['FromStopID'] == 5]
        distance = distance.loc[distance['ToStopID'] == 5]
        print(distance)
        edge1 = g[8][8]
        print(edge1)
    except:
        print('No Node from 8 -> 8')


def draw_graph(graph):
    nx.draw(graph, with_labels=True, node_size=60,font_size=6, arrowsize=1)
    plt.savefig('plots/1draw_network-{}.png'.format(datetime.now().strftime("%H:%M:%S")), dpi=1200)
    plt.clf()

    nx.draw_networkx(graph, with_labels=True, node_size=60,font_size=6, arrowsize=1)
    plt.savefig('plots/2draw_networkx_network-{}.png'.format(datetime.now().strftime("%H:%M:%S")), dpi=1200)
    plt.clf()

    #nx.draw_networkx_nodes(graph, 8)
    #plt.savefig('plots/draw_networkx_nodes_network-{}.png'.format(datetime.now().strftime("%H:%M:%S")), dpi=900)

    #nx.draw_networkx_edges(graph)
    #plt.savefig('plots/draw_networkx_edges_network-{}.png'.format(datetime.now().strftime("%H:%M:%S")), dpi=900)

    #nx.draw_networkx_labels(graph)
    #plt.savefig('plots/draw_networkx_labels_network-{}.png'.format(datetime.now().strftime("%H:%M:%S")), dpi=900)

    #nx.draw_networkx_edge_labels(graph)
    #plt.savefig('plots/draw_networkx_edge_labels_network-{}.png'.format(datetime.now().strftime("%H:%M:%S")), dpi=900)

    nx.draw_circular(graph, with_labels=True, node_size=60,font_size=6, arrowsize=1)
    plt.savefig('plots/3draw_circular_network-{}.png'.format(datetime.now().strftime("%H:%M:%S")), dpi=1200)
    plt.clf()

    nx.draw_kamada_kawai(graph, with_labels=True, node_size=60,font_size=6, arrowsize=1)
    plt.savefig('plots/4network-{}.png'.format(datetime.now().strftime("%H:%M:%S")), dpi=1200)
    plt.clf()

    nx.draw_random(graph, with_labels=True, node_size=60,font_size=6, arrowsize=1)
    plt.savefig('plots/5draw_random_network-{}.png'.format(datetime.now().strftime("%H:%M:%S")), dpi=1200)
    plt.clf()

    nx.draw_kamada_kawai(graph, with_labels=True, node_size=60,font_size=6, arrowsize=1)
    plt.savefig('plots/6draw_random_network-{}.png'.format(datetime.now().strftime("%H:%M:%S")), dpi=1200)
    plt.clf()

    nx.draw_spectral(graph, with_labels=True,node_size=60,font_size=6, arrowsize=1)
    plt.savefig('plots/7draw_spectral_network-{}.png'.format(datetime.now().strftime("%H:%M:%S")), dpi=1200)
    plt.clf()

    nx.draw_spring(graph, with_labels=True, node_size=60,font_size=6, arrowsize=1)
    plt.savefig('plots/8draw_spring_network-{}.png'.format(datetime.now().strftime("%H:%M:%S")), dpi=1200)
    plt.clf()

    nx.draw_shell(graph, with_labels=True, node_size=60,font_size=6, arrowsize=1)
    plt.savefig('plots/9draw_spring_network-{}.png'.format(datetime.now().strftime("%H:%M:%S")), dpi=1200)
    plt.clf()

    #nx.draw_planar(graph, with_labels=True)
    #plt.savefig('plots/draw_planar_network-{}.png'.format(datetime.now().strftime("%H:%M:%S")), dpi=900)
    #plt.clf()

