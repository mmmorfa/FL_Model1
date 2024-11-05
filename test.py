import numpy as np
import pandas as pd
from copy import deepcopy
from random import randint
from math import log2, ceil, floor
import sqlite3
import json
import datetime



# Pandas config
pd.set_option("display.max_rows", None, "display.max_columns", None)

# ****************************** VNF Generator GLOBALS ******************************
# File directory
db_name = '/home/mario/Documents/Joint RAN-MEC Slicing & RA O-RAN/gym_examples/slice_request_db{}'.format(datetime.datetime.now())
DIRECTORY = db_name

# Number of VNF types dictionary
# i.e. {key: value}, value = [MEC_CPU, MEC_RAM, MEC_Storage, MEC_BW, RAN_R, RAN_L]
VNF_TYPES = {0: [2, 4, 10, 8, 15, 7], 1: [4, 8, 32, 20, 45, 1], 2: [4, 8, 20, 14, 30, 1], 3: [2, 4, 8, 5, 10, 8], 
             4: [4, 12, 64, 30, 50, 15], 5: [1, 2, 5, 2, 8, 35], 6: [2, 4, 10, 10, 5, 40], 7: [4, 16, 64, 30, 54, 20], 
             8: [2, 16, 20, 25, 35, 25], 9: [4, 8, 25, 15, 40, 30], 10: [4, 4, 16, 20, 25, 2], 11: [4, 8, 16, 25, 28, 5]}
# Arrival rates from VNF types dictionary
ARRIVAL_RATE = {0: 3, 1: 2, 2: 3, 3: 4, 4: 2, 5: 3, 6: 3, 7: 2, 8: 3, 9: 4, 10: 2, 11: 3}
# VNF life cycle from VNF types dictionary
LIFE_CYCLE_RATE = {0: 10, 1: 8, 2: 5, 3: 3, 4: 9, 5: 10, 6: 10, 7: 8, 8: 5, 9: 3, 10: 9, 11: 10}
# Num of vnf requests
NUM_VNF_REQUESTS = 100

# ****************************** VNF Generator FUNCTIONS ******************************

def generate_requests_per_type(key, num):
    """ This function generates a set of requests per type """
    req = []
    vnf_request_at_time = 0

    x = 0  # to check the inter arrival times
    y = 0  # to check the holding times

    for _ in range(num):
        # Generate inter-arrival time for the VNF request
        inter_vnf_time_request = np.random.exponential(1.0 / ARRIVAL_RATE[key])
        # Update the time for the next request
        vnf_request_at_time += inter_vnf_time_request

        # Generate holding time for the VNF request
        #vnf_request_life_time = np.random.exponential(LIFE_CYCLE_RATE[key])
        # Alternative: Use a Poisson distribution for holding time
        vnf_request_life_time = np.random.poisson(LIFE_CYCLE_RATE[key]) 
        vnf_kill_at_time = vnf_request_at_time + vnf_request_life_time

        final_vnf = [vnf_request_at_time, VNF_TYPES[key][0],VNF_TYPES[key][1],VNF_TYPES[key][2],VNF_TYPES[key][3],VNF_TYPES[key][4],VNF_TYPES[key][5], vnf_kill_at_time]
        #final_vnf = [vnf_request_at_time, VNF_TYPES[key][0], vnf_kill_at_time]

        # Round up decimals
        final_vnf = [round(val, 3) if isinstance(val, (int, float)) else val for val in final_vnf]
        req.append(final_vnf)

        x += inter_vnf_time_request
        y += vnf_request_life_time

    # print("DEBUG: key = ", key, "average inter-arrival = ", x / num, "average holding = ", y / num)
    return req


def get_key(val):
    """ Get value key """
    for k, v in VNF_TYPES.items():
        if val == v:
            return k

def generate_vnf_list():
    # ****************************** MAIN CODE ******************************
    # The overall procedure to create the requests is as follows:
        # - generate a set of requests per type
        # - put them altogether
        # - sort them according the arrival time
        # - return the num_VNFs_requests number of them '''

    vnfList = []

    for vnf in list(VNF_TYPES.values()):
        # Get vnf key for the arrival and holding dicts
        key = get_key(vnf)

            # We don't know how many requests from each type will be in the final list of requests.
            # It depends on the arrival rate of the type in comparison to the rates of the other types.
            # So, we generate the maximum number, i.e., num_VNFs_requests.
        requests = generate_requests_per_type(key, NUM_VNF_REQUESTS)

            # vnfList will be all the requests from all types not sorted.
        for req in requests:
            vnfList.append(req)

    # Sort the requests according to the arrival rate
    vnfList.sort(key=lambda x: x[0])

        # Until now, we have generated num_VNFs_requests * len(vnf_types) requests.
        # We only need the num_VNFs_requests of them'''
    vnfList = vnfList[:NUM_VNF_REQUESTS]

        # Dataframe
    columns = ['ARRIVAL_REQUEST_@TIME','SLICE_MEC_CPU_REQUEST', 'SLICE_MEC_RAM_REQUEST', 'SLICE_MEC_STORAGE_REQUEST', 'SLICE_MEC_BW_REQUEST', 'SLICE_RAN_R_REQUEST','SLICE_RAN_L_REQUEST', 'SLICE_KILL_@TIME']
    df = pd.DataFrame(data=vnfList, columns=columns, dtype=float)

        # Export df to  csv file
    df.to_csv(DIRECTORY, index=False, header=True)

generate_vnf_list()