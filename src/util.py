import numpy as np
import random

STEP_NUM = 1
PRINT_EVERY = 1

assert PRINT_EVERY <= STEP_NUM

def normalize_data(data, method="max_min"):
    data = np.array(data)
    if method == "max_min":
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    elif method == "sum":
        if len(data.shape) > 1:
            ### assume data's shape is (client_num, task_num)
            return data / np.sum(data, axis=0)
        else:
            return data / np.sum(data)
    else:
        raise ValueError(method)

def calcualte_client_value(price_table, client_feature_list):
    ''' price_table is a 2-D list, shape=(#client, #task)
    price_table[client_id][task_id] is the price of client_id for task_id
    '''
    idle_cost_list = [feature[1] for feature in client_feature_list]
    value_table = []
    for client_idx in range(len(client_feature_list)):
        client_price_list = price_table[client_idx]
        value_list = [price / (idle_cost_list[client_idx]+1) for price in  client_price_list]
        value_table.append(value_list)
    return value_table

def sigmoid(data):
    return 1/ (1+ np.exp(-data))

def remove_list_indexed(removed_ele, original_l, ll):
    new_original_l = []
    for i in original_l:
        new_original_l.append(i)
    for i in new_original_l:
        if i == removed_ele:
            new_original_l.remove(i)
    for i in range(len(ll)):
        if set(ll[i]) == set(new_original_l):
            return i
    return -1

def shapley_list_indexed(original_l, ll):
    for i in range(len(ll)):
        if set(ll[i]) == set(original_l):
            return i
    return -1

def PowerSetsBinary(items):
    N = len(items)
    set_all = []
    for i in range(2 ** N):
        combo = []
        for j in range(N):
            if (i >> j) % 2 == 1:
                combo.append(items[j])
        set_all.append(combo)
    return set_all

def sample_config(l, id, use_random=True):
    if use_random:
        return random.sample(l, 1)[0]
    else:
        return l[id%len(l)]

class bcolors:
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    CBLINK = '\33[5m'
    CBLINK2 = '\33[6m'
    CSELECTED = '\33[7m'

    CBLACK  = '\33[30m'
    CRED    = '\33[31m'
    CGREEN  = '\33[32m'
    CYELLOW = '\33[33m'
    CBLUE   = '\33[34m'
    CVIOLET = '\33[35m'
    CBEIGE  = '\33[36m'
    CWHITE  = '\33[37m'

    CBLACKBG  = '\33[40m'
    CREDBG    = '\33[41m'
    CGREENBG  = '\33[42m'
    CYELLOWBG = '\33[43m'
    CBLUEBG   = '\33[44m'
    CVIOLETBG = '\33[45m'
    CBEIGEBG  = '\33[46m'
    CWHITEBG  = '\33[47m'

    FAIL = '\33[31m'
    WARNING = '\33[33m'