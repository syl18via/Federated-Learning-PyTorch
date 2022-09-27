import copy
import itertools
import time
from tqdm import tqdm
from scipy.special import comb

def calculate_sv_v1(models, model_evaluation_func, averaging_func):
    """
    Computes the Shapley Value for clients

    Parameters:
    models (dict): Key value pair of client identifiers and model updates.
    model_evaluation_func (func) : Function to evaluate model update.
    averaging_func (func) : Function to used to average the model updates.

    Returns:
    sv: Key value pair of client identifiers and the computed shapley values.

    """

    # generate possible permutations
    all_perms = list(itertools.permutations(list(models.keys())))
    marginal_contributions = []
    # history map to avoid retesting the models
    history = {}

    pbar = tqdm(total=(len(all_perms) * len(models)))
    for perm in all_perms:
        # print(perm)
        perm_values = {}
        local_models = {}

        for client_id in perm:
            model = copy.deepcopy(models[client_id])
            local_models[client_id] = model

            # get the current index eg: (A,B,C) on the 2nd iter, the index is (A,B)
            if len(perm_values.keys()) == 0:
                index = (client_id,)
            else:
                index = tuple(sorted(list(tuple(perm_values.keys()) + (client_id,))))

            if index in history.keys():
                current_value = history[index]
            else:
                # print(f"Evaluate {list(local_models.keys())}")
                model = averaging_func(local_models)
                current_value = model_evaluation_func(model)
                history[index] = current_value

            # print(f"perm_values[{client_id}] =  V({list(local_models.keys())}) - V({list(perm_values.keys())})")
            perm_values[client_id] =  current_value - sum(perm_values.values())
            pbar.update(1)

        marginal_contributions.append(perm_values)

    pbar.close()
    sv = {client_id: 0 for client_id in models.keys()}

    # sum the marginal contributions
    for perm in marginal_contributions:
        for key, value in perm.items():
            sv[key] += value

    # compute the average marginal contribution
    sv = {key: value / len(marginal_contributions) for key, value in sv.items()}
    return sv

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

def remove_list_indexed(removed_ele, current_set, all_sets):
    new_set = []
    for i in current_set:
        if i != removed_ele:
            new_set.append(i)
    for i in range(len(all_sets)):
        if set(all_sets[i]) == set(new_set):
            return i
    return -1

def calculate_sv_v2(models, model_evaluation_func, averaging_func):
    ### Calculate the Feedback
    selected_client_num = len(models)
    all_sets = PowerSetsBinary(list(models.keys()))
    group_shapley_value = []

    for s in tqdm(all_sets):
        if len(s) == 0:
            current_value = 0
        else:
            local_models = dict([(client_id, models[client_id]) for client_id in s])
            model = averaging_func(local_models)
            current_value = model_evaluation_func(model)
        group_shapley_value.append(current_value)

    agent_shapley = []
    for index in range(selected_client_num):
        shapley = 0.0
        for set_idx, j in enumerate(all_sets):
            if index in j:
                remove_list_index = remove_list_indexed(index, j, all_sets)
                if remove_list_index != -1:
                    shapley += (group_shapley_value[set_idx] - group_shapley_value[
                        remove_list_index]) / (comb(selected_client_num - 1, len(all_sets[remove_list_index])))

        agent_shapley.append(shapley)
    # for ag_s in agent_shapley:
    #     print(ag_s)
    # task.select_clients(agent_shapley, free_client)
    # if sum(agent_shapley) == 0:
    #     import code
    #     code.interact(local=locals())
    return agent_shapley


def calculate_sv(models, model_evaluation_func, averaging_func):
    ts = time.time()
    # sv = calculate_sv_v1(models, model_evaluation_func, averaging_func)
    sv = calculate_sv_v2(models, model_evaluation_func, averaging_func)
    print(f"Take {time.time()-ts:.3f} s to calculate sv for {list(models.keys())}")