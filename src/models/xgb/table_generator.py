# THIS FILE IS PART OF Planter PROJECT
# Planter.py - The core part of the Planter library
#
# THIS PROGRAM IS FREE SOFTWARE TOOL, WHICH MAPS MACHINE LEARNING ALGORITHMS TO DATA PLANE, IS LICENSED UNDER Apache-2.0
# YOU SHOULD HAVE RECEIVED A COPY OF WTFPL LICENSE, IF NOT, PLEASE CONTACT THE FOLLOWING E-MAIL ADDRESSES
#
# Copyright (c) 2020-2021 Changgang Zheng
# Copyright (c) Computing Infrastructure Lab, Department of Engineering Science, University of Oxford
# E-mail: changgang.zheng@eng.ox.ac.uk or changgangzheng@qq.com
#
# Functions: This file is responsible for training, algorithm mapping, and software testing of the ML model.
#            Please refer to ./Docs/Planter_User_Document.pdf or further information.

from src.functions.Range_to_TCAM_Top_Down import Table_to_TCAM
from src.functions.json_encoder import NpEncoder
from eval.eval_metrics import eval_metrics
from pathos.multiprocessing import ProcessingPool
from datetime import datetime
import numpy as np
import xgboost as xgb
import os
import copy
import math
import time
import json

def map(value):
    value = value
    return value

def get_path(model, conditions, path, num, leaf_info, tree_index):
    if 'children' in model.keys():
        conditions_yes = copy.deepcopy(conditions)
        conditions_no = copy.deepcopy(conditions)
        if conditions_yes[model["split"]][1] > map(model["split_condition"])-1:
            conditions_yes[model["split"]][1] = map(model["split_condition"])-1
        if conditions_no[model["split"]][0] < map(model["split_condition"]) :
            conditions_no[model["split"]][0] = map(model["split_condition"])
        for child_model in model["children"]:
            if child_model["nodeid"]==model["yes"]:
                path, num, leaf_info = get_path(child_model, conditions_yes, path, num, leaf_info, tree_index)
            if child_model["nodeid"]==model["no"]:
                path, num, leaf_info = get_path(child_model, conditions_no, path, num, leaf_info, tree_index)
    else:
        # print(path, conditions)
        path['path '+str(num)] = conditions
        path['path '+str(num)]['leaf'] = model["leaf"]
        # leaf_info['tree '+str(tree_index)] += [model["leaf"]]
        leaf_info['tree ' + str(tree_index)] += [round(model["leaf"], 1)]
        if model["leaf"] > leaf_info['max value']: leaf_info['max value'] = model["leaf"]
        elif model["leaf"] < leaf_info['min value']: leaf_info['min value'] = model["leaf"]
        num += 1
    return path, num, leaf_info

def find_feature_split(model, tree_index, num_features, feature_names):
    count_layer = 0
    count_route = 0
    count_list = 0
    layer = {}
    route = {}
    layer[count_layer] = {}
    layer[count_layer][count_list] = {}
    layer[count_layer][count_list]["lst"] = [0]
    layer[count_layer][count_list]["tab"] = model
    feature_split = {}
    num_features = len(feature_names)

    for i in range(num_features):
        feature_split["feature " + str(i)] = []
    while True:
        if len(layer[count_layer].keys()) == 0:
            break
        layer[count_layer + 1] = {}
        count_list = 0
        for list_id in layer[count_layer]:
            feature_split["feature " + str(feature_names.index(layer[count_layer][list_id]["tab"]["split"]))] += [
                layer[count_layer][list_id]["tab"]["split_condition"]]
            # (optional add -1)The -1 means the feature splits is for <= =, so each split is largest value in each range

            for i, children in enumerate(layer[count_layer][list_id]["tab"]["children"]):
                if "children" not in children.keys():
                    route[count_route] = layer[count_layer][list_id]["lst"] + [children["nodeid"]]
                    count_route += 1
                else:
                    layer[count_layer + 1][count_list] = {}
                    layer[count_layer + 1][count_list]["lst"] = layer[count_layer][list_id]["lst"] + [
                        children["nodeid"]]
                    layer[count_layer + 1][count_list]["tab"] = children
                    count_list += 1
        count_layer += 1
    for f in range(num_features):
        feature_split['feature ' + str(f)] = sorted(list(set(feature_split['feature ' + str(f)])))
    return feature_split

def generate_feature_tables(split, num_features,feature_max, table):
    for i in range(num_features):
        table["feature "+str(i)] = {}
        count_code = 0
        nife = sorted(split["feature "+str(i)])
        for j in range(feature_max[i]+1):
            if nife !=[] :
                if len(nife) > count_code:
                    if j == nife[count_code]:
                        count_code+=1
            table["feature " + str(i)][j] = count_code
    return table

def path_to_path_to_leaf(path, num_features, table, leaf_code_list):
    path_to_leaf ={}
    for p in path:
        path_to_leaf[p] = {}
        path_to_leaf[p]['leaf'] = leaf_code_list.index(round(path[p]['leaf'], 1))
        for f in range(num_features):
            ini = table['feature '+str(f)][path[p]['f'+str(f)][0]]
            end = table['feature '+str(f)][path[p]['f'+str(f)][1]]
            path_to_leaf[p]['feature '+str(f)] = np.arange(ini,end+1).tolist()
    return path_to_leaf

def find_path_for_leaf_nodes(model, feature_split, feature_max, num_features, table, leaf_info, tree_index):
    conditions = {}
    for i in range(num_features):
        conditions["f" + str(i)] = [0, feature_max[i]]
        feature_split["feature " + str(i)] += [feature_max[i]]

    path = {}
    path, _, leaf_info = get_path(model, conditions, path, 0, leaf_info, tree_index)
    leaf_info['tree '+str(tree_index)] = sorted(list(set(leaf_info['tree '+str(tree_index)])))
    path_to_leaf = path_to_path_to_leaf(path, num_features, table, leaf_info['tree '+str(tree_index)] )
    return path_to_leaf, leaf_info

def generate_code_table_for_path(table, leaf_path, code_dict, feature_num, num_features, count):
    if feature_num == num_features:
        table['code to vote'][count] = {}
        for f in range(num_features):
            table['code to vote'][count]['f'+str(f)+' code'] = code_dict['feature ' + str(f)]
        table['code to vote'][count]['leaf'] = leaf_path['leaf']
        count += 1
        return table, count
    else:
        for value in leaf_path['feature '+str(feature_num)]:
            code_dict['feature ' + str(feature_num)] = value
            feature_num += 1
            table, count = generate_code_table_for_path(table, leaf_path, code_dict, feature_num, num_features, count)
            feature_num -= 1
    return table, count

def generate_code_table(table, path_to_leaf, num_features):
    table['code to vote'] = {}
    count = 0
    for p in path_to_leaf:
        table, count = generate_code_table_for_path(table, path_to_leaf[p], {}, 0, num_features, count)
    return table

def generate_table(model,tree_index, g_table, num_features, feature_names, feature_max, leaf_info):

    feature_split = find_feature_split(model, tree_index, num_features, feature_names)
    g_table[tree_index] = {}
    g_table[tree_index] = generate_feature_tables(feature_split, num_features, feature_max, g_table[tree_index])
    leaf_info['tree '+str(tree_index)] = []
    path_to_leaf, leaf_info = find_path_for_leaf_nodes(model, feature_split, feature_max, num_features, g_table[tree_index], leaf_info, tree_index)

    code_width_for_feature = np.zeros(num_features)
    for i in range(num_features):
        code_width_for_feature[i] = np.ceil(math.log(
            g_table[tree_index]['feature ' + str(i)][np.max(list(g_table[tree_index]['feature ' + str(i)].keys()))] + 1, 2))
    g_table[tree_index] = generate_code_table(g_table[tree_index], path_to_leaf, num_features)
    print('\rThe table for Tree: {} is generated'.format(tree_index), end="")
    return g_table, leaf_info

def ten_to_bin(num,count):
    num = bin(int(num)).lstrip('0b')

    if len(num) != count:
        cont = count - len(num)
        num = cont * '0' + num
    return num

def MaxMin_Norm_with_range(x, min , max, ranges = 10):
    """[0,1] normaliaztion"""
    x = (x - min) / (max - min)
    return np.floor(ranges*x)

def run_model(train_X, train_y, test_X, test_y, used_features, cur_dataset,
              cur_trace, config_path=None):
    if config_path:
        print(f'Config: {config_path}')
        config = json.load(open(config_path, 'r'))
    else:
        config = json.load(open('conf/planter_config.json', 'r'))

    num_features        = config['data config']['number of features']
    cur_model           = config['model config']['model']
    model_size          = config['model config']['model size']
    num_classes         = config['model config']['number of classes']
    num_boost_rounds    = int(int(config['model config']['number of trees'])/config['model config']['number of classes'])
    # num_trees           = config['model config']['number of classes'] * int(int(config['model config']['number of trees']) / config['model config']['number of classes'])
    num_trees           = int(int(config['model config']['number of trees']) / config['model config']['number of classes'])
    num_depth           = config['model config']['number of depth']
    max_leaf_nodes      = config['model config']['max number of leaf nodes']

    feature_names = []
    for i, f in enumerate(used_features):
        train_X.rename(columns={f: "f"+str(i)}, inplace=True)
        test_X.rename(columns={f: "f" + str(i)}, inplace=True)
        feature_names+=["f"+str(i)]

    feature_max = []
    for i in feature_names:
        t_t = [test_X[[i]].max()[0], train_X[[i]].max()[0]]
        feature_max += [np.max(t_t)+1]

    # =================== train model timer ===================
    config['timer log']['train model'] = {}
    config['timer log']['train model']['start'] = time.time()
    # =================== train model timer ===================

    # XGBoost

    data_train = xgb.DMatrix(train_X, label=train_y)
    data_test = xgb.DMatrix(test_X, label=test_y)
    watchlist = [(data_test, 'eval'), (data_train, 'train')]
    # param = {'max_depth': num_depth, 'eta': 1, 'silent': 0, 'objective': 'binary:logistic'}
    # param = {'max_depth': num_depth, 'eta': 1, 'silent': 0, 'objective': 'multi:softmax', 'num_class': num_classes}
    param = {'max_depth': num_depth, 'eta': 1, 'silent': 0, 'objective': 'multi:softprob', 'num_class': num_classes}
    bst = xgb.train(param, data_train, num_boost_round=num_boost_rounds, evals=watchlist)

    # param = {'max_depth': 8, 'num_class': 2}
    # bst = xgb.train(param, data_train, num_boost_round=200, evals=watchlist)
    bst.dump_model("src/tmp/tree.txt")

    sklearn_y_predict_probas = bst.predict(data_test)
    # sklearn_y_predict       = np.where(sklearn_y_predict_proba > 0.5, 1, 0)
    sklearn_y_predict       = np.argmax(sklearn_y_predict_probas, axis=1)
    # sklearn_y_predict_proba = sklearn_y_predict_probas[np.arange(sklearn_y_predict_probas.shape[0]), sklearn_y_predict]
    sklearn_y_predict_proba = sklearn_y_predict_probas[np.arange(sklearn_y_predict_probas.shape[0]), 1]

    # print(f'sklearn predict_probas: {sklearn_y_predict_probas}')
    # print(f'sklearn predict: {sklearn_y_predict}')
    # print(f'sklearn predict_proba: {sklearn_y_predict_proba}')

    eval_metrics(test_y,
                 sklearn_y_predict,
                 sklearn_y_predict_proba,
                 cur_dataset,
                 cur_trace,
                 cur_model,
                 model_size,
                 'sklearn')

    # =================== train model timer ===================
    config['timer log']['train model']['end'] = time.time()
    # =================== train model timer ===================

    # =================== convert model timer ===================
    config['timer log']['convert model'] = {}
    config['timer log']['convert model']['start'] = time.time()
    # =================== convert model timer ===================

    log_file = 'src/logs/log.json'
    if os.path.exists(log_file):
        log_dict = json.load(open(log_file, 'r'))
    else:
        log_dict = {}

    if ("num_feature: " + str(num_features)) not in log_dict:
        log_dict["num_feature: " + str(num_features)] = {}
    if ("num_tree: " + str(num_trees)) not in log_dict["num_feature: " + str(num_features)]:
        log_dict["num_feature: " + str(num_features)]["num_tree: " + str(num_trees)] = {}
    if ("num_depth: " + str(num_depth)) not in log_dict["num_feature: " + str(num_features)][
        "num_tree: " + str(num_trees)]:
        log_dict["num_feature: " + str(num_features)]["num_tree: " + str(num_trees)][
            "num_depth: " + str(num_depth)] = {}
    # log_dict["num_feature: " + str(num_features)]["num_tree: " + str(num_trees)]["num_depth: " + str(num_depth)][
        # "classification_report"] = result
    log_dict["num_feature: " + str(num_features)]["num_tree: " + str(num_trees)]["num_depth: " + str(num_depth)][
        "max number of leaf nodes"] = max_leaf_nodes
    json.dump(log_dict, open(log_file, 'w'), indent=4)
    print('Classification results are downloaded to log as', log_file)

    the_model= bst.get_dump(fmap="", with_stats=False, dump_format="json")
    xgb_model = {}
    for i, m in enumerate(the_model):
        xgb_model[i] = json.loads(m)

    g_table = {}
    leaf_info ={}
    leaf_info['max value'] = 0
    leaf_info['min value'] = 0
    for idx in xgb_model:
        estimator = xgb_model[idx]
        g_table, leaf_info = generate_table(estimator, idx, g_table, num_features, feature_names, 
                                            feature_max, leaf_info)

    def votes_to_class(tree_num, vote_list, num_trees, num_classes, g_table, num, leaf_info):
        if tree_num  == num_trees:
            vote = np.zeros(num_classes).tolist()
            for t in range(num_trees):
                vote[t%num_classes] += leaf_info["tree "+str(t)][vote_list[t]]
            # if vote.index(np.max(vote))== 0:
            # if True :
            g_table['votes to class'][num] = {}
            for t in range(len(vote_list)):
                g_table['votes to class'][num]['t'+str(t)+' vote'] = vote_list[t]
            g_table['votes to class'][num]['class'] = vote.index(np.max(vote))
            num += 1
            return g_table, num
        else:
            for value in range(len(leaf_info["tree "+str(tree_num)])):
                vote_list[tree_num] = value
                tree_num += 1
                g_table, num = votes_to_class(tree_num, vote_list, num_trees, num_classes, g_table, num, leaf_info)
                tree_num -= 1
        return g_table, num

    ranges = 10
    g_table['votes to class'] = {}
    print("\nGenerating vote to class table...",end="")
    g_table, _ = votes_to_class(0, np.zeros(num_trees).tolist(), num_trees, num_classes, g_table, 0, leaf_info)
    print('Done')

    feature_width = []
    for maxs in feature_max:
        feature_width += [int(np.ceil(math.log(maxs, 2)) + 1)]

    code_width_tree_feature = np.zeros((num_trees,num_features))
    for i in range(num_features):
        for tree in range(num_trees):
            code_width_tree_feature[tree, i] = np.ceil(math.log(g_table[tree]['feature ' + str(i)][np.max(list(g_table[tree]['feature ' + str(i)].keys()))]+1,2)+1)

    Ternary_Table = {}
    Ternary_Table['decision'] = g_table['votes to class']

    for tree in range(num_trees):
        Ternary_Table['tree ' + str(tree)] = g_table[tree]['code to vote']

    for i in range(num_features):
        Ternary_Table['feature '+str(i)] = {}
        for value in range(feature_max[i]):
            Ternary_Table['feature ' + str(i)][value] = []
            for tree in range(num_trees):
                Ternary_Table['feature ' + str(i)][value] += [g_table[tree]["feature "+str(i)][value]]
    Exact_Table = copy.deepcopy(Ternary_Table)
    for i in range(num_features):
        if i!=0:
            print('')
        print('Begine transfer: Feature table ' + str(i))
        Ternary_Table['feature '+str(i)]= Table_to_TCAM(Ternary_Table['feature '+str(i)], feature_width[i])

    # ===================== prepare default vote =========================
    print("\nPreparing default vote...", end="")
    collect_votes = []
    for t in range(num_trees):
        for idx in Exact_Table['tree ' + str(t)]:
            collect_votes += [int(Exact_Table['tree ' + str(t)][idx]['leaf'])]
    default_vote = max(collect_votes, key=collect_votes.count)

    code_table_size = 0
    for t in range(num_trees):
        Ternary_Table['tree ' + str(t)] = {}
        for idx in Exact_Table['tree ' + str(t)]:
            if int(Exact_Table['tree ' + str(t)][idx]['leaf']) != default_vote:
                Ternary_Table['tree ' + str(t)][code_table_size] = Exact_Table['tree ' + str(t)][idx]
                code_table_size += 1
        Exact_Table['tree ' + str(t)] = copy.deepcopy(Ternary_Table['tree ' + str(t)])
    print('Done')

    # ===================== prepare default class =========================
    print("Preparing default class...", end="")
    collect_class = np.zeros(num_classes).tolist()
    for idx in Exact_Table['decision']:
        collect_class[Exact_Table['decision'][idx]['class']] += 1
    default_class = collect_class.index(max(collect_class))

    code_table_size = 0
    Ternary_Table['decision'] = {}
    for idx in Exact_Table['decision']:
        if Exact_Table['decision'][idx]['class'] != default_class:
            Ternary_Table['decision'][code_table_size] = Exact_Table['decision'][idx]
            code_table_size += 1
    Exact_Table['decision'] = copy.deepcopy(Ternary_Table['decision'])
    print('Done')

    # =================== convert model timer ===================
    config['timer log']['convert model']['end'] = time.time()
    # =================== convert model timer ===================

    table_name = 'ternary_table.json'
    json.dump(Ternary_Table, open(f'eval/tables/{cur_dataset}/{cur_model}/{cur_trace}-{cur_model}-'
                                  f'{model_size}-{table_name}', 'w'), indent=4)
    print('\ntable_ternary is generated')
    table_name = 'exact_table.json'
    json.dump(Exact_Table, open(f'eval/tables/{cur_dataset}/{cur_model}/{cur_trace}-{cur_model}-'
                                f'{model_size}-{table_name}', 'w'), indent=4)

    config['p4 config'] = {}
    config['p4 config']["model"] = "XGB"
    config['p4 config']["number of features"] = num_features
    config['p4 config']["number of classes"] = num_classes
    config['p4 config']["number of trees"] = num_trees
    config['p4 config']['table name']           = f'{cur_trace}-{cur_model}-{model_size}-ternary_table.json'
    config['p4 config']["decision table size"] = len(Ternary_Table['decision'].keys())
    config['p4 config']["code table size"] = []
    for tree in range(num_trees):
        config['p4 config']["code table size"] += [len(Ternary_Table['tree ' + str(tree)].keys())]
    config['p4 config']["default vote"] = default_vote
    config['p4 config']["default label"] = default_class
    config['p4 config']["width of feature"] = feature_width
    config['p4 config']["width of code"] = code_width_tree_feature
    config['p4 config']["used columns"] = []
    for i in range(num_features):
        config['p4 config']["used columns"] += [len(Ternary_Table['feature ' + str(i)].keys())]
    config['p4 config']["width of probability"] = 7
    config['p4 config']["width of result"] = 8
    config['p4 config']["standard headers"] = ["ethernet", "Planter", "arp", "ipv4", "tcp", "udp", "vlan_tag"]
    config['test config'] = {}
    config['test config']['type of test'] = 'classification'

    json.dump(config,
              open(config['directory config']['work']+"/"+config_path, 'w'),
              indent=4,
              cls=NpEncoder)

    # main()
    return sklearn_y_predict.tolist()

def process_packets(batch, test_X, num_trees, num_features, Ternary_Table, Exact_Table,
                    config, test_y, sklearn_test_y):
    results = []
    core_id = os.getpid()
    for i in batch:

        vote_list = np.zeros(num_trees).astype(dtype=int).tolist()
        input_feature_value = test_X.values[i]

        for tree in range(num_trees):
            code_list = np.zeros(num_features)
            ternary_code_list = np.zeros(num_features)

            for f in range(num_features):
                match_or_not = False

                # match ternary
                TCAM_table = Ternary_Table['feature ' + str(f)]
                keys = list(TCAM_table.keys())

                for count in keys:
                    if input_feature_value[f] & TCAM_table[count][0] == TCAM_table[count][0] & TCAM_table[count][1]:
                        ternary_code_list[f] = TCAM_table[count][2][tree]
                        match_or_not = True
                        break

                if not match_or_not:
                    print('feature table not matched')
                # match exact
                code_list[f] = Exact_Table['feature ' + str(f)][str(input_feature_value[f])][tree]
                if not match_or_not:
                    print('feature table not matched')

            if str(code_list) != str(ternary_code_list):
                print('error in exact to ternary match', code_list, ternary_code_list)

            for key in Exact_Table["tree " + str(tree)]:
                match_or_not = False
                all_True = True
                for code_f in range(num_features):
                    if not Exact_Table["tree " + str(tree)][key]['f' + str(code_f) + ' code'] == code_list[code_f]:
                        all_True = False
                        break
                if all_True:
                    vote_list[tree] = int(Exact_Table["tree " + str(tree)][key]['leaf'])
                    match_or_not = True
                    break
            if not match_or_not:
                vote_list[tree] = config['p4 config']["default vote"]

        switch_prediction = config['p4 config']["default label"]
        for key in Exact_Table['decision']:
            match_or_not = False
            all_True = True
            for tree_v in range(num_trees):
                if not Exact_Table["decision"][key]['t' + str(tree_v) + ' vote'] == vote_list[tree_v]:
                    all_True = False
                    break
            if all_True:
                switch_prediction = Exact_Table['decision'][key]['class']
                match_or_not = True
                break

        switch_proba = sum(vote_list) / len(vote_list)
        if switch_prediction == 0:
            switch_proba = 1 - switch_proba

        results.append((switch_prediction, switch_proba))

    cur_ts = datetime.now()
    cur_ts = cur_ts.strftime("%H-%M-%S")
    print(f'[{cur_ts}]  Core {core_id} processed {len(batch)} packets.')
    return results

def test_tables(sklearn_test_y, test_X, test_y, cur_dataset, cur_trace,
                config_path=None, threshold=None):
    if config_path:
        print(config_path)
        config = json.load(open(config_path, 'r'))
    else:
        config = json.load(open('conf/planter_config.json', 'r'))

    num_features    = config['data config']['number of features']
    cur_model       = config['model config']['model']
    model_size      = config['model config']['model size']
    num_trees       = int(int(config['model config']['number of trees'])/config['model config']['number of classes'])

    Ternary_Table   = json.load(open(f'eval/tables/{cur_dataset}/{cur_model}/{cur_trace}-'
                                     f'{cur_model}-{model_size}-ternary_table.json', 'r'))
    Exact_Table     = json.load(open(f'eval/tables/{cur_dataset}/{cur_model}/{cur_trace}-'
                                     f'{cur_model}-{model_size}-exact_table.json', 'r'))

    print('Test the exact feature table, extact code and decision table (feel free if the acc to sklearn is slightly lower than 1)')

    same = 0
    correct = 0
    error = 0
    switch_test_y = []
    switch_test_y_proba = []
    test_y_new = []

    batch_size = 10000

    indices = list(range(np.shape(test_X.values)[0]))
    batches = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]

    with ProcessingPool() as pool:
        results = pool.map(lambda batch: process_packets(batch, test_X, num_trees, num_features, Ternary_Table, Exact_Table, config, test_y, sklearn_test_y), batches)

        for batch_results in results:
            for i, (switch_prediction, switch_proba) in enumerate(batch_results):
                switch_test_y.append(switch_prediction)
                switch_test_y_proba.append(switch_proba)

                # Calculate correct, same, and error counts
                original_index = batches[i // batch_size][i % batch_size]  # Get the original index
                test_y_new.append(test_y[original_index])

                if switch_prediction == test_y[original_index]:
                    correct += 1

                if switch_prediction == sklearn_test_y[original_index]:
                    same += 1
                else:
                    error += 1

    eval_metrics(test_y_new,
                 switch_test_y,
                 switch_test_y_proba,
                 cur_dataset,
                 cur_trace,
                 cur_model,
                 model_size,
                 'switch')

def resource_prediction(config_path):
    config = json.load(open(config_path, 'r'))

    print('Exact match entries:     ', np.sum(config['p4 config']["code table size"]) \
          + config['p4 config']["decision table size"] )
    print('Ternary match entries:   ', np.sum(config['p4 config']["used columns"]))
