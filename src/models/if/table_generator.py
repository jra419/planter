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

from sklearn.ensemble import IsolationForest
from src.functions.json_encoder import NpEncoder
from src.functions.Range_to_TCAM_Top_Down import *
from src.functions.Range_to_LPM import Table_to_LPM
from src.functions.Muti_Exact_to_LPM import *
from eval.eval_metrics import eval_metrics
from sklearn.tree import _tree
from pathos.multiprocessing import ProcessingPool as Pool
from datetime import datetime
import os
import numpy as np
import math
import json
import copy
import re

def get_lineage(tree, feature_names, file):
    left            = tree.tree_.children_left
    right           = tree.tree_.children_right
    threshold       = tree.tree_.threshold
    features        = [feature_names[i] for i in tree.tree_.feature]
    value           = tree.tree_.value
    n_node_samples  = tree.tree_.n_node_samples

    le  = '<='
    g   = '>'

    # get ids of child nodes
    idx = np.argwhere(left == -1)[:, 0]

    # traverse the tree and get the node information
    def recurse(left, right, child, lineage=None):
        if lineage is None:
            lineage = [child]
        if child in left:
            parent  = np.where(left == child)[0].item()
            split   = 'l'
        else:
            parent  = np.where(right == child)[0].item()
            split   = 'r'

        lineage.append((parent, split, threshold[parent], features[parent]))

        if parent == 0:
            lineage.reverse()
            return lineage
        else:
            return recurse(left, right, parent, lineage)

    for j, child in enumerate(idx):
        clause = ' when '

        for node in recurse(left, right, child):
            if len(str(node)) < 3:
                continue
            i = node

            if not isinstance(i, tuple):
                continue

            if i[1] == 'l':
                sign = le
            else:
                sign = g

            clause = clause + i[3] + sign + str(i[2]) + ' and '

        # write the node information into text file
        # print(node)

        ind     = n_node_samples[node]
        clause  = clause[:-4] + ' then ' + str(ind)

        file.write(clause)
        file.write(";\n")

def print_tree(tree, feature_names):
    tree_           = tree.tree_
    feature_name    = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    # print('feature_name:', feature_name)

    print("def tree({}):".format(", ".join(feature_names)))
    share = {}

    def recurse(node, depth, share):
        indent = "  " * depth

        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name        = feature_name[node]
            share[name] = {}
            threshold   = tree_.threshold[node]

            print("{}if {} <= {}:".format(indent, name, threshold))

            recurse(tree_.children_left[node], depth + 1, share)

            print("{}else:  # if {} > {}".format(indent, name, threshold))

            recurse(tree_.children_right[node], depth + 1, share)
        else:
            print("{}return {}".format(indent, tree_.value[node]))

    recurse(0, 1, share)

def ten_to_bin(num, count):
    num = bin(int(num)).lstrip('0b')
    if len(num) != count:
        cont = count - len(num)
        num = cont * '0' + num
    return num

def find_feature_split(model, tree_index, num_features):
    feature_names = []
    feature_split = {}

    for a in range(num_features):
        feature_split["feature "+str(a)] = []
        feature_names += ["f" + chr(ord('A') + a)]

    threshold   = model.tree_.threshold
    features    = [feature_names[i] for i in model.tree_.feature]

    for i, fe in enumerate(features):
        for a in range(num_features):
            if a == 0:
                if fe == feature_names[a]:
                    feature_split["feature "+str(a)].append(threshold[i])
                    continue
            if fe == feature_names[a]:
                if threshold[i] != -2.0:
                    feature_split["feature "+str(a)].append(threshold[i])
                continue

    for a in range(num_features):
        feature_split["feature "+str(a)] = \
                [int(np.floor(i)) for i in feature_split["feature "+str(a)]]
        feature_split["feature "+str(a)].sort()

    tree = open('src/tmp/tree'+str(tree_index)+'.txt', "w+")

    for a in range(num_features):
        tree.write(str(feature_names[a]) + " = ")
        tree.write(str(feature_split["feature "+str(a)]))
        tree.write(";\n")

    # print_tree(model, feature_names)

    get_lineage(model, feature_names, tree)
    tree.close()

    action      = [0, 1]
    textfile    = 'src/tmp/tree'+str(tree_index)+'.txt'

    for f in range(num_features):
        feature_split['feature ' + str(f)] = sorted(list(set(feature_split['feature ' + str(f)])))

    return textfile, feature_split

def generate_feature_tables(split, num_features,feature_max, table):
    for i in range(num_features):
        table["feature "+str(i)]    = {}
        count_code                  = 0
        nife                        = sorted(split["feature "+str(i)])

        for j in range(feature_max[i]+1):
            if nife !=[] :
                if len(nife) > count_code:
                    if j-1 == nife[count_code]:
                        count_code+=1
            table["feature " + str(i)][j] = count_code
    return table

def find_classification(textfile, feature_split, num_features):
    fea         = []
    sign        = []
    num         = []
    f           = open(textfile, 'r')
    feature_n   = {}
    text        = r"("

    for b in range(num_features):
        feature_n[b] = []

        if b == 0:
            text += "f"+chr(ord('A')+b)
        else:
            text += "|f" + chr(ord('A')+b)

    text += ")"

    for line in f:
        n = re.findall(r"when", line)
        if n:
            fea.append(re.findall(text, line))
            sign.append(re.findall(r"(<=|>)", line))
            num.append(re.findall(r"\d+\.?\d*", line))

    f.close()

    classfication   = []
    featuren        = {}

    for i in range(len(fea)):
        num_nodes = 0

        for b in range(num_features):
            featuren[b] = [k for k in range(len(feature_split["feature "+str(b)]) + 1)]

        for j, feature in enumerate(fea[i]):
            for b in range(num_features):
                if feature == "f"+chr(ord('A')+b):
                    sig         = sign[i][j]
                    thres       = int(float(num[i][j]))
                    id          = feature_split["feature "+str(b)].index(thres)
                    num_nodes  += 1

                    if sig == '<=':
                        while id < len(feature_split["feature "+str(b)]):
                            if id + 1 in featuren[b]:
                                featuren[b].remove(id + 1)
                            id = id + 1
                    else:
                        while id >= 0:
                            if id in featuren[b]:
                                featuren[b].remove(id)
                            id = id - 1
                    continue

        for b in range(num_features):
            feature_n[b].append(featuren[b])

        a = len(num[i])

        classfication.append(num_nodes)

    return feature_n, classfication

def find_path_for_leaf_nodes(feature_n, classfication, num_features):
    path_to_leaf = {}

    for i in range(len(classfication)):
        path_to_leaf["path "+str(i)] = {}
        path_to_leaf["path " + str(i)]["leaf"] = classfication[i]

        for j in range(num_features):
            path_to_leaf["path " + str(i)]["feature "+str(j)] = feature_n[j][i]

    return path_to_leaf

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
            table, count = generate_code_table_for_path(table, leaf_path, code_dict,
                                                        feature_num, num_features, count)
            feature_num -= 1

    return table, count

def generate_code_table(table, path_to_leaf, num_features):
    table['code to vote']   = {}
    count                   = 0

    for p in path_to_leaf:
        table, count = generate_code_table_for_path(table, path_to_leaf[p], {}, 0,
                                                    num_features, count)

    return table

def generate_table(model, tree_index, num_features, g_table, feature_max, leaf_info):
    textfile, feature_split = find_feature_split(model, tree_index, num_features)

    g_table[tree_index] = {}
    g_table[tree_index] = generate_feature_tables(feature_split, num_features,
                                                  feature_max, g_table[tree_index])

    feature_n, classfication    = find_classification(textfile, feature_split , num_features)
    path_to_leaf                = find_path_for_leaf_nodes(feature_n, classfication, num_features)
    code_width_for_feature      = np.zeros(num_features)

    for i in range(num_features):
        code_width_for_feature[i] = int(np.ceil(math.log(g_table[tree_index]['feature ' + str(i)][np.max(list(g_table[tree_index]['feature ' + str(i)].keys()))]+1,2))) or 1

    g_table[tree_index] = generate_code_table(g_table[tree_index], path_to_leaf, num_features)

    print('\rThe table for Tree: {} is generated'.format(tree_index), end="")

    leaf_info['tree '+str(tree_index)]= np.unique(classfication)

    return g_table, leaf_info

def votes_to_class(tree_num, vote_list, num_trees, num_classes, g_table, num,
                   leaf_info, path_len_threshold):
    if tree_num  == num_trees:
        vote = 0

        for t in range(num_trees):
            vote += leaf_info["tree "+str(t)][vote_list[t]]
        g_table['votes to class'][num] = {}

        for t in range(len(vote_list)):
            g_table['votes to class'][num]['t'+str(t)+' vote'] = \
                    leaf_info["tree "+str(t)][vote_list[t]]

        if vote >= path_len_threshold*num_trees:
            g_table['votes to class'][num]['class'] = 0
        else:
            g_table['votes to class'][num]['class'] = 1

        num += 1

        return g_table, num
    else:
        for value in range(len(leaf_info["tree "+str(tree_num)])):
            vote_list[tree_num]     = value
            tree_num               += 1
            g_table, num = votes_to_class(tree_num, vote_list, num_trees, num_classes,
                                          g_table, num, leaf_info, path_len_threshold)
            tree_num               -= 1
    return g_table, num

def run_model(train_X, train_y, test_X, test_y, used_features, cur_dataset,
              cur_trace, config_path=None):
    if config_path:
        print(f'Config: {config_path}')
        config = json.load(open(config_path, 'r'))
    else:
        config = json.load(open('conf/planter_config.json', 'r'))

    num_features    = config['data config']['number of features']
    cur_model       = config['model config']['model']
    model_size      = config['model config']['model size']
    num_samples     = config['model config']['number of samples']
    num_classes     = config['model config']['number of classes']
    num_trees       = config['model config']['number of trees']

    path_len_threshold = (2 * (np.log(num_samples - 1) + np.euler_gamma) - (2 * (num_samples - 1) / num_samples)) * (-math.log(0.6, 2))

    print("The threshold of path length is %.2f" % path_len_threshold)

    feature_names = []
    for i, f in enumerate(used_features):
        train_X.rename(columns={f: "f" + str(i)}, inplace=True)
        test_X.rename(columns={f: "f" + str(i)}, inplace=True)
        feature_names += ["f" + str(i)]

    feature_max = []
    for i in feature_names:
        t_t = [test_X[[i]].max()[0], train_X[[i]].max()[0]]
        feature_max += [np.max(t_t)+1]

    rng = np.random.RandomState(42)

    # fit the model
    clf = IsolationForest(n_estimators= num_trees,
                          max_samples=num_samples,
                          random_state=rng)
    clf.fit(train_X)

    y_pred_test         = clf.predict(test_X)
    sklearn_y_predict   = copy.deepcopy(y_pred_test)
    sklearn_y_scores    = (-1.0) * clf.decision_function(test_X)

    for i in range(len(y_pred_test)):
        if y_pred_test[i] == -1:
            sklearn_y_predict[i] = 1
        if y_pred_test[i] == 1:
            sklearn_y_predict[i] = 0

    eval_metrics(test_y,
                 sklearn_y_predict,
                 sklearn_y_scores,
                 cur_dataset,
                 cur_trace,
                 cur_model,
                 model_size,
                 'sklearn')

    g_table                 = {}
    leaf_info               = {}
    leaf_info['max value']  = 0
    leaf_info['min value']  = 0

    for idx, estimator in enumerate(clf.estimators_):
        g_table, leaf_info = generate_table(estimator, idx, num_features, g_table,
                                            feature_max, leaf_info)

    g_table['votes to class'] = {}
    print("\nGenerating vote to class table...", end="")
    g_table, _ = votes_to_class(0, np.zeros(num_trees).tolist(), num_trees, num_classes,
                                g_table, 0, leaf_info, path_len_threshold)
    print('Done')

    feature_width = []
    for max_f in feature_max:
        feature_width += [int(np.ceil(math.log(max_f, 2)) + 1)]

    code_width_tree_feature = np.zeros((num_trees, num_features))

    for i in range(num_features):
        for tree in range(num_trees):
            code_width_tree_feature[tree, i] = int(np.ceil(math.log(
                g_table[tree]['feature ' + str(i)][np.max(list(g_table[tree]['feature ' + str(i)].keys()))] + 1, 2) + 1)) or 1

    LPM_Table               = {}
    LPM_Table['decision']   = g_table['votes to class']

    for tree in range(num_trees):
        LPM_Table['tree ' + str(tree)] = g_table[tree]['code to vote']

    for i in range(num_features):
        LPM_Table['feature ' + str(i)] = {}

        for value in range(feature_max[i]):
            LPM_Table['feature ' + str(i)][value] = []

            for tree in range(num_trees):
                LPM_Table['feature ' + str(i)][value] += \
                        [g_table[tree]["feature " + str(i)][value]]

    Exact_Table = copy.deepcopy(LPM_Table)

    for i in range(num_features):
        if i != 0:
            print('')
        print('Begin transfer: Feature table ' + str(i))

        LPM_Table['feature ' + str(i)] = Table_to_LPM(LPM_Table['feature ' + str(i)], feature_width[i])

    # ===================== prepare default vote =========================

    collect_votes = []
    for t in range(num_trees):
        for idx in Exact_Table['tree ' + str(t)]:
            collect_votes += [int(Exact_Table['tree ' + str(t)][idx]['leaf'])]

    default_vote = max(collect_votes, key=collect_votes.count)

    code_table_size = 0
    for t in range(num_trees):
        LPM_Table['tree ' + str(t)] = {}

        for idx in Exact_Table['tree ' + str(t)]:
            if int(Exact_Table['tree ' + str(t)][idx]['leaf']) != default_vote:
                LPM_Table['tree ' + str(t)][code_table_size] = \
                        Exact_Table['tree ' + str(t)][idx]
                code_table_size += 1
        Exact_Table['tree ' + str(t)] = copy.deepcopy(LPM_Table['tree ' + str(t)])

    # ===================== prepare default class =========================

    collect_class = []

    for idx in Exact_Table['decision']:
        collect_class += [Exact_Table['decision'][idx]['class']]

    default_class = max(collect_class, key=collect_class.count)

    code_table_size = 0

    LPM_Table['decision'] = {}
    for idx in Exact_Table['decision']:
        if Exact_Table['decision'][idx]['class'] != default_class:
            LPM_Table['decision'][code_table_size] = Exact_Table['decision'][idx]
            code_table_size += 1

    Exact_Table['decision'] = copy.deepcopy(LPM_Table['decision'])

    json.dump(LPM_Table, open(f'eval/tables/{cur_dataset}/{cur_model}/{cur_trace}-{cur_model}-'
                              f'{model_size}-lpm_table.json', 'w'), indent=4, cls=NpEncoder)

    json.dump(Exact_Table, open(f'eval/tables/{cur_dataset}/{cur_model}/{cur_trace}-{cur_model}-'
                                f'{model_size}-exact_table.json', 'w'), indent=4, cls=NpEncoder)

    config['p4 config']                         = {}
    config['p4 config']["model"]                = "if"
    config['p4 config']["number of features"]   = num_features
    config['p4 config']["number of classes"]    = num_classes
    config['p4 config']["number of trees"]      = num_trees
    config['p4 config']['table name']           = f'{cur_trace}-{cur_model}-{model_size}-lpm_table.json'
    config['p4 config']["decision table size"]  = len(LPM_Table['decision'].keys())
    config['p4 config']["code table size"]      = []
    for tree in range(num_trees):
        config['p4 config']["code table size"] += [len(LPM_Table['tree ' + str(tree)].keys())]
    config['p4 config']["default vote"]         = default_vote
    config['p4 config']["default label"]        = default_class
    config['p4 config']["width of feature"]     = feature_width
    config['p4 config']["width of code"]        = code_width_tree_feature
    config['p4 config']["used columns"]         = []
    for i in range(num_features):
        config['p4 config']["used columns"]    += [len(LPM_Table['feature ' + str(i)].keys())]
    config['p4 config']["width of probability"] = 7
    config['p4 config']["width of result"]      = 8
    config['p4 config']["standard headers"]     = ["ethernet", "Planter", "arp", "ipv4", "tcp",
                                                   "udp", "vlan_tag"]
    config['test config']                       = {}
    config['test config']['type of test']       = 'classification'

    json.dump(config,
              open(config['directory config']['work']+"/"+config_path, 'w'),
              indent=4,
              cls=NpEncoder)

    return sklearn_y_predict.tolist()

def process_batch(batch_indices, test_X, num_trees, num_features, table_lpm, table_exact,
                  config):
    results = []
    core_id = os.getpid()

    for i in batch_indices:
        vote_list           = np.zeros(num_trees).astype(dtype=int).tolist()
        anomaly_cnt         = 0
        input_feature_value = test_X.values[i]

        for tree in range(num_trees):
            code_list           = np.zeros(num_features)
            lpm_code_list       = np.zeros(num_features)
            input_feature_value = test_X.values[i]

            for f in range(num_features):
                match_or_not = False

                # match ternary
                LPM_table   = table_lpm['feature ' + str(f)]
                keys        = list(LPM_table.keys())
                mask        = []
                action      = []

                # For each value in LPM table, check if it matches that separation key
                for count in np.sort(keys):
                    # if there is a ternary match
                    if input_feature_value[f] & LPM_table[count][0] == LPM_table[count][0] & LPM_table[count][1]:
                        mask.append(LPM_table[count][0])
                        action.append(LPM_table[count][2])

                max_mask            = max(mask)
                max_index           = mask.index(max_mask)
                # Choose the action with the longest prefix match
                lpm_code_list[f]    = action[max_index][tree]

                # match exact
                code_list[f] = table_exact['feature ' + str(f)][str(input_feature_value[f])][tree]

            if str(code_list) != str(lpm_code_list):
                print('error in exact to ternary match', code_list, lpm_code_list)

            for key in table_exact["tree " + str(tree)]:
                match_or_not    = False
                all_True        = True

                for code_f in range(num_features):
                    if not table_exact["tree " + str(tree)][key]['f' + str(code_f) + ' code'] == code_list[code_f]:
                        all_True = False
                        break

                if all_True:
                    vote_list[tree] = int(table_exact["tree " + str(tree)][key]['leaf'])
                    match_or_not    = True
                    break

            if not match_or_not:
                vote_list[tree] = config['p4 config']["default vote"]

        for key in table_exact['decision']:
            match_or_not    = False
            all_True        = True

            for tree_v in range(num_trees):
                if not table_exact["decision"][key]['t' + str(tree_v) + ' vote'] == vote_list[tree_v]:
                    all_True = False
                    break
                else:
                    anomaly_cnt += 1

            if all_True:
                switch_prediction = table_exact['decision'][key]['class']
                match_or_not = True
                break

        if not match_or_not:
            switch_prediction = config['p4 config']["default label"]

        # results.append((switch_prediction, (-1.0) * sum(vote_list) / len(vote_list)))
        results.append((switch_prediction, anomaly_cnt / num_trees))

    cur_ts = datetime.now()
    cur_ts = cur_ts.strftime("%H-%M-%S")
    print(f'[{cur_ts}]  Core {core_id} processed {len(batch_indices)} packets.')
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
    num_trees       = config['model config']['number of trees']

    LPM_Table       = json.load(open(f'eval/tables/{cur_dataset}/{cur_model}/{cur_trace}-'
                                     f'{cur_model}-{model_size}-lpm_table.json', 'r'))
    Exact_Table     = json.load(open(f'eval/tables/{cur_dataset}/{cur_model}/{cur_trace}-'

                                     f'{cur_model}-{model_size}-exact_table.json', 'r'))

    print('Test the exact feature table, extact code and decision table (feel free if the acc to sklearn is slightly lower than 1)')

    switch_test_y       = []
    switch_test_y_proba = []

    batch_size = 10000

    indices = list(range(np.shape(test_X.values)[0]))
    batches = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]

    with Pool() as pool:
        results = pool.map(lambda batch: process_batch(batch, test_X, num_trees, num_features, LPM_Table, Exact_Table, config), batches)

        for batch_results in results:
            for i, (switch_prediction, switch_proba) in enumerate(batch_results):
                switch_test_y.append(switch_prediction)
                switch_test_y_proba.append(switch_proba)

    eval_metrics(test_y,
                 switch_test_y,
                 switch_test_y_proba,
                 cur_dataset,
                 cur_trace,
                 cur_model,
                 model_size,
                 'switch')

# SINGLE PROCESS

# def test_tables(sklearn_test_y, test_X, test_y, cur_dataset, cur_trace,
#                 config_path=None, threshold=None):
#     if config_path:
#         print(config_path)
#         config = json.load(open(config_path, 'r'))
#     else:
#         config = json.load(open('conf/planter_config.json', 'r'))

#     num_features    = config['data config']['number of features']
#     cur_model       = config['model config']['model']
#     model_size      = config['model config']['model size']
#     num_trees       = config['model config']['number of trees']

#     LPM_Table       = json.load(open(f'eval/tables/{cur_dataset}/{cur_model}/{cur_trace}-'
#                                      f'{cur_model}-{model_size}-lpm_table.json', 'r'))
#     Exact_Table     = json.load(open(f'eval/tables/{cur_dataset}/{cur_model}/{cur_trace}-'

#                                      f'{cur_model}-{model_size}-exact_table.json', 'r'))

#     print('Test the exact feature table, extact code and decision table (feel free if the acc to sklearn is slightly lower than 1)')

#     switch_test_y       = []
#     switch_test_y_proba = []

#     for i in range(np.shape(test_X.values)[0]):
#         vote_list = np.zeros(num_trees).astype(dtype=int).tolist()

#         for tree in range(num_trees):
#             code_list           = np.zeros(num_features)
#             lpm_code_list       = np.zeros(num_features)
#             input_feature_value = test_X.values[i]

#             for f in range(num_features):
#                 match_or_not = False

#                 # match ternary
#                 LPM_table   = LPM_Table['feature ' + str(f)]
#                 keys        = list(LPM_table.keys())
#                 mask        = []
#                 action      = []

#                 # For each value in LPM table, check if it matches that separation key
#                 for count in np.sort(keys):
#                     # if there is a ternary match
#                     if input_feature_value[f] & LPM_table[count][0] == LPM_table[count][0] & LPM_table[count][1]:
#                         mask.append(LPM_table[count][0])
#                         action.append(LPM_table[count][2])

#                 max_mask            = max(mask)
#                 max_index           = mask.index(max_mask)
#                 # Choose the action with the longest prefix match
#                 lpm_code_list[f]    = action[max_index][tree]

#                 # match exact
#                 code_list[f] = Exact_Table['feature ' + str(f)][str(input_feature_value[f])][tree]

#             if str(code_list) != str(lpm_code_list):
#                 print('error in exact to ternary match', code_list, lpm_code_list)

#             for key in Exact_Table["tree " + str(tree)]:
#                 match_or_not    = False
#                 all_True        = True

#                 for code_f in range(num_features):
#                     if not Exact_Table["tree " + str(tree)][key]['f' + str(code_f) + ' code'] == code_list[code_f]:
#                         all_True = False
#                         break

#                 if all_True:
#                     vote_list[tree] = int(Exact_Table["tree " + str(tree)][key]['leaf'])
#                     match_or_not    = True
#                     break

#             if not match_or_not:
#                 vote_list[tree] = config['p4 config']["default vote"]

#         for key in Exact_Table['decision']:
#             match_or_not    = False
#             all_True        = True

#             for tree_v in range(num_trees):
#                 if not Exact_Table["decision"][key]['t' + str(tree_v) + ' vote'] == vote_list[tree_v]:
#                     all_True = False
#                     break

#             if all_True:
#                 switch_prediction = Exact_Table['decision'][key]['class']
#                 match_or_not = True
#                 break

#         if not match_or_not:
#             switch_prediction = config['p4 config']["default label"]


#         switch_test_y_proba += [ (-1.0) * sum(vote_list)/len(vote_list) ]
#         switch_test_y       += [switch_prediction]

#     eval_metrics(test_y,
#                  switch_test_y,
#                  switch_test_y_proba,
#                  cur_dataset,
#                  cur_trace,
#                  cur_model,
#                  model_size,
#                  'switch')
