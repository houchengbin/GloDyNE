import numpy as np
import networkx as nx
import datetime
import os
import pickle

"""
The dataset contains 733 daily instances which span an interval of 785 days from November 8 1997 to January 2 2000

We take the latest 15 dates from 19991217 to 19991231:

@ graph 0 # of nodes 2070 # of edges 4596
@ graph 1 # of nodes 2073 # of edges 4604
@ graph 2 # of nodes 2102 # of edges 4675
@ graph 3 # of nodes 2067 # of edges 4580
@ graph 4 # of nodes 2080 # of edges 4638
@ graph 5 # of nodes 2095 # of edges 4659
@ graph 6 # of nodes 2083 # of edges 4629
@ graph 7 # of nodes 2058 # of edges 4587
@ graph 8 # of nodes 2063 # of edges 4594
@ graph 9 # of nodes 2092 # of edges 4653
@ graph 10 # of nodes 2089 # of edges 4637
@ graph 11 # of nodes 2122 # of edges 4707
@ graph 12 # of nodes 2120 # of edges 4683
@ graph 13 # of nodes 2132 # of edges 4715
@ graph 14 # of nodes 2107 # of edges 4676
"""


def detect_exentence_file(date):
    file_location = date_2_string(date)
    return os.path.isfile(file_location)

def date_2_string(date):#change the date format to
    year = date.year
    month = date.month
    day = date.day
    string_date = str(year*10000 + month*100 + day)
    string_date = 'as'+string_date+'.txt'
    return string_date

def string_2_date(date_str):
    year = date_str[0:4]
    month = date_str[4:6]
    day = date_str[6:8]
    date = datetime.datetime(year = int(year), month = int(month), day = int(day))
    return date


def generate_dynamic_graph(start_date = '19991009', time_step_number = 10, stop_at_irregular_interval = False):
    """
    earlist date is 19971108
    last date is 20000102

    the form of input is a string of date such as '19991015'. Note that I did not implement any date check

    :param start_date:
    :param time_step_number:
    :param stop_at_irregular_interval:
    :return:
    """
    user_chosen_date = string_2_date(start_date)
    dyanmic_netowks = []
    last_available_date = datetime.datetime(2000,1,2)


    remaining_graph = time_step_number
    while(remaining_graph > 0):
        if (user_chosen_date - last_available_date).days > 0:
            print("no more file available, stop generate more file")
            break
        elif detect_exentence_file(user_chosen_date) == True:

            remaining_graph -= 1
            file_name = date_2_string(user_chosen_date)
            graph = generate_a_graph(file_name)
            dyanmic_netowks.append(graph)
            print(remaining_graph)
            #print(len(graph.nodes())," graph node number")
            user_chosen_date += datetime.timedelta(days=1)
        elif stop_at_irregular_interval == False:
            print("file does not exit at ", user_chosen_date, "date skipped")
            user_chosen_date += datetime.timedelta(days=1)
        else:
            print("file does not exit at ", user_chosen_date, "stop generate more network")
            break
    print("dynamic network length:",len(dyanmic_netowks))
    return dyanmic_netowks



def generate_a_graph(file_name):

    graph_data = np.genfromtxt(file_name, dtype=str)
    graph = nx.Graph()

    for i in range(len(graph_data)):
        graph.add_edge(graph_data[i][0],graph_data[i][1])

    return graph

def save_nx_graph(nx_graph, path='nx_graph_temp.data'):
    with open(path, 'wb') as f:
        pickle.dump(nx_graph, f, protocol=pickle.HIGHEST_PROTOCOL)  # the higher protocol, the smaller file
    with open(path, 'rb') as f:
        nx_graph_reload = pickle.load(f)

    try:
        print('Check if it is correctly dumped and loaded: ', nx_graph_reload.edges() == nx_graph.edges(),
              ' It contains only ONE graph')
    except:
        for i in range(len(nx_graph)):
            print('Check if it is correctly dumped and loaded: ', nx_graph_reload[i].edges() == nx_graph[i].edges(),
                  ' for Graph ', i)

if __name__ == '__main__':
    """
    start_date = datetime.datetime(1997,11,8)
    last_date = datetime.datetime(2000,1,2)
    days = (last_date-start_date).days
    print(days)
    #785 internal however which 733 days are missing
    print(date_2_string(start_date))
    detect_exentence_file()
    """
    graphs = generate_dynamic_graph(start_date='19991217', time_step_number=15, stop_at_irregular_interval=False)
    save_nx_graph(nx_graph=graphs, path='as733_dyn_graphs.pkl')

    graphs = graphs[-15:]
    for i in range(len(graphs)):
        print('@ graph', i, '# of nodes', len(graphs[i].nodes()), '# of edges', len(graphs[i].edges()))
