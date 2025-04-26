# THIS FILE IS PART OF Planter PROJECT
# Planter.py - The core part of the Planter library
#
# THIS PROGRAM IS FREE SOFTWARE TOOL, WHICH MAPS MACHINE LEARNING ALGORITHMS TO DATA PLANE, IS LICENSED UNDER Apache-2.0
# YOU SHOULD HAVE RECEIVED A COPY OF THE LICENSE, IF NOT, PLEASE CONTACT THE FOLLOWING E-MAIL ADDRESSES
#
# Copyright (c) 2020-2021 Changgang Zheng
# Copyright (c) Computing Infrastructure Lab, Department of Engineering Science, University of Oxford
# E-mail: changgang.zheng@eng.ox.ac.uk or changgangzheng@qq.com
#
# Functions: This file is responsible for training, algorithm mapping, and software testing of the ML model.
#            Please refer to ./Docs/Planter_User_Document.pdf or further information.

from sklearn.tree import _tree
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
from src.functions.Range_to_TCAM_Top_Down import Table_to_TCAM
from src.functions.json_encoder import NpEncoder
from eval.eval_metrics import eval_metrics
import numpy as np
import time
import math
import re
import json
import copy
import os

ts_datetime = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')[:-3]

def get_lineage(tree, feature_names, file):
    left = tree.tree_.children_left
    right = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features = [feature_names[i] for i in tree.tree_.feature]
    value = tree.tree_.value
    le = '<='
    g = '>'
    # get ids of child nodes
    idx = np.argwhere(left == -1)[:, 0]
    # traverse the tree and get the node information
    def recurse(left, right, child, lineage=None):
        if lineage is None:
            lineage = [child]
        if child in left:
            parent = np.where(left == child)[0].item()
            split = 'l'
        else:
            parent = np.where(right == child)[0].item()
            split = 'r'
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
        # wirte the node information into text file
        a = list(value[node][0])
        ind = a.index(np.max(a))
        clause = clause[:-4] + ' then ' + str(ind)
        file.write(clause)
        file.write(";\n")

def print_tree(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    # print('feature_name:', feature_name)
    print("def tree({}):".format(", ".join(feature_names)))
    share = {}
    def recurse(node, depth, share):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            share[name] = {}
            threshold = tree_.threshold[node]
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
        first_letter = int(np.floor(a / 24))
        second_letter = int(a % 24)
        feature_names += ["f" +chr(ord('A')+first_letter)+chr(ord('A')+second_letter)]
    threshold = model.tree_.threshold
    features = [feature_names[i] for i in model.tree_.feature]
    for i, fe in enumerate(features):
        for b in range(num_features):
            if b == 0:
                if fe == feature_names[b]:
                    feature_split["feature "+str(b)].append(threshold[i])
                    continue
            if fe == feature_names[b]:
                if threshold[i] != -2.0:
                    feature_split["feature "+str(b)].append(threshold[i])
                continue
    for c in range(num_features):
        feature_split["feature "+str(c)] = [int(np.floor(i)) for i in feature_split["feature "+str(c)]]
        feature_split["feature "+str(c)].sort()
    tree = open('src/tmp/tree'+str(tree_index)+'.txt', "w+")
    for d in range(num_features):
        tree.write(str(feature_names[d]) + " = ")
        tree.write(str(feature_split["feature "+str(d)]))
        tree.write(";\n")
    # print_tree(model, feature_names)
    get_lineage(model, feature_names, tree)
    tree.close()
    textfile = 'src/tmp/tree'+str(tree_index)+'.txt'
    for f in range(num_features):
        feature_split['feature ' + str(f)] = sorted(list(set(feature_split['feature ' + str(f)])))
    return textfile, feature_split

def generate_feature_tables(split, num_features,feature_max, table):
    for i in range(num_features):
        table["feature "+str(i)] = {}
        count_code = 0
        nife = sorted(split["feature "+str(i)])
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

    for a in range(num_features):
        feature_n[a]    = []
        first_letter    = int(np.floor(a / 24))
        second_letter   = int(a % 24)
        if a == 0:
            text += "f"+chr(ord('A')+first_letter)+chr(ord('A')+second_letter)
        else:
            text += "|f"+chr(ord('A')+first_letter)+chr(ord('A')+second_letter)
    text += ")"

    for line in f:
        n = re.findall(r"when", line)
        if n:
            fea.append(re.findall(text, line))
            sign.append(re.findall(r"(<=|>)", line))
            num.append(re.findall(r"\d+\.?\d*", line))
    f.close()

    classification  = []
    features        = {}

    for i in range(len(fea)):
        for a in range(num_features):
            features[a] = [k for k in range(len(feature_split["feature "+str(a)]) + 1)]
        for j, feature in enumerate(fea[i]):
            for b in range(num_features):
                first_letter    = int(np.floor(b / 24))
                second_letter   = int(b % 24)
                if feature == "f"+chr(ord('A')+first_letter)+chr(ord('A')+second_letter):
                    sig     = sign[i][j]
                    thres   = int(float(num[i][j]))
                    id      = feature_split["feature "+str(b)].index(thres)
                    if sig == '<=':
                        while id < len(feature_split["feature "+str(b)]):
                            if id + 1 in features[b]:
                                features[b].remove(id + 1)
                            id = id + 1
                    else:
                        while id >= 0:
                            if id in features[b]:
                                features[b].remove(id)
                            id = id - 1
                    continue
        for c in range(num_features):
            feature_n[c].append(features[c])
        a = len(num[i])
        classification.append(num[i][a - 1])

    return feature_n, classification

def find_path_for_leaf_nodes(feature_n, classification, num_features):
    path_to_leaf = {}
    for i in range(len(classification)):
        path_to_leaf["path "+str(i)] = {}
        path_to_leaf["path " + str(i)]["leaf"] = classification[i]
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
    table['code to vote'] = {}
    count = 0
    for p in path_to_leaf:
        table, count = generate_code_table_for_path(table, path_to_leaf[p], {}, 0,
                                                    num_features, count)
    return table

def generate_table(model, tree_index, num_features, g_table, feature_max):
    textfile, feature_split     = find_feature_split(model, tree_index, num_features)
    g_table[tree_index]         = {}
    g_table[tree_index]         = generate_feature_tables(feature_split, num_features,
                                                          feature_max, g_table[tree_index])
    feature_n, classfication    = find_classification(textfile, feature_split , num_features)
    path_to_leaf                = find_path_for_leaf_nodes(feature_n, classfication, num_features)
    code_width_for_feature      = np.zeros(num_features)

    for i in range(num_features):
        code_width_for_feature[i] = int(np.ceil(math.log(g_table[tree_index]['feature ' + str(i)][np.max(list(g_table[tree_index]['feature ' + str(i)].keys()))]+1,2))) or 1
    g_table[tree_index] = generate_code_table(g_table[tree_index], path_to_leaf, num_features)
    print('\rThe table for Tree: {} is generated'.format(tree_index), end="")
    return g_table

def votes_to_class(tree_num, vote_list, num_trees, num_classes, g_table, num):
    if tree_num  == num_trees:
        vote = np.zeros(num_classes).tolist()
        for i in range(num_trees):
            vote[vote_list[i]] += 1
        g_table['votes to class'][num] = {}
        for t in range(len(vote_list)):
            g_table['votes to class'][num]['t'+str(t)+' vote'] = vote_list[t]
        g_table['votes to class'][num]['class'] = vote.index(np.max(vote))
        num += 1
        return g_table, num
    else:
        for value in range(num_classes):
            vote_list[tree_num] = value
            tree_num += 1
            g_table, num = votes_to_class(tree_num, vote_list, num_trees, num_classes, g_table, num)
            tree_num -= 1
    return g_table, num

def run_model(train_X, train_y, test_X, test_y, used_features, cur_dataset,
              cur_trace, config_path=None):
    if config_path:
        print(f'Config: {config_path}')
        config = json.load(open(config_path, 'r'))
    else:
        config = json.load(open('conf/planter_config.json', 'r'))

    config['model config']['number of classes'] = int(np.max(train_y) + 1)

    num_features    = config['data config']['number of features']
    num_classes     = config['model config']['number of classes']
    num_depth       = config['model config']['number of depth']
    num_trees       = config['model config']['number of trees']
    max_leaf_nodes  = config['model config']['max number of leaf nodes']

    feature_names = []
    for i, f in enumerate(used_features):
        train_X.rename(columns={f: "f" + str(i)}, inplace=True)
        test_X.rename(columns={f: "f" + str(i)}, inplace=True)
        feature_names += ["f" + str(i)]

    feature_max = []
    for i in feature_names:
        t_t = [test_X[[i]].max()[0], train_X[[i]].max()[0]]
        feature_max += [np.max(t_t)+1]

    # =================== train model timer ===================
    config['timer log']['train model'] = {}
    config['timer log']['train model']['start'] = time.time()
    # =========================================================

    # Random Forest

    rfc = RandomForestClassifier(n_estimators=num_trees,
                                 max_depth=num_depth,
                                 max_leaf_nodes=max_leaf_nodes)
    rfc.fit(train_X, train_y)

    sklearn_y_predict           = rfc.predict(test_X)
    # sklearn_y_predict_proba     = rfc.predict_proba(test_X)
    sklearn_y_predict_proba   = rfc.predict_proba(test_X)[:,1]

    eval_metrics(test_y,
                 sklearn_y_predict,
                 sklearn_y_predict_proba,
                 f'rf-{num_trees}-{num_depth}-{max_leaf_nodes}-sklearn',
                 cur_dataset,
                 cur_trace)

    # result = classification_report(test_y, sklearn_y_predict, digits= 4)

    # =================== train model timer ===================
    config['timer log']['train model']['end'] = time.time()
    # =========================================================

    # =================== convert model timer ===================
    config['timer log']['convert model']            = {}
    config['timer log']['convert model']['start']   = time.time()
    # ===========================================================

    # exit()
    log_file = 'src/logs/log.json'
    if os.path.exists(log_file):
        log_dict = json.load(open(log_file, 'r'))
    else:
        log_dict = {}

    if ( "num_feature: "+str(num_features)) not in log_dict:
        log_dict["num_feature: "+str(num_features)] = {}
    if ( "num_tree: "+str(num_trees)) not in log_dict["num_feature: "+str(num_features)]:
        log_dict["num_feature: "+str(num_features)]["num_tree: "+str(num_trees)] = {}
    if ( "num_depth: "+str(num_depth)) not in log_dict["num_feature: "+str(num_features)]["num_tree: "+str(num_trees)]:
        log_dict["num_feature: "+str(num_features)]["num_tree: "+str(num_trees)]["num_depth: "+ str(num_depth)]= {}
    # log_dict["num_feature: " + str(num_features)][ "num_tree: " + str(num_trees)]["num_depth: " + str(num_depth)]["classification_report"] = result
    log_dict["num_feature: " + str(num_features)][ "num_tree: " + str(num_trees)]["num_depth: " + str(num_depth)]["max number of leaf nodes"] =max_leaf_nodes
    json.dump(log_dict, open(log_file, 'w'), indent=4)
    print ('Classification results are downloaded to log as', log_file)

    g_table = {}
    for idx, estimator in enumerate(rfc.estimators_):
        g_table = generate_table(estimator, idx,  num_features ,g_table, feature_max)

    print("\nGenerating vote to class table...", end="")
    g_table['votes to class'] = {}
    g_table, _ = votes_to_class(0, np.zeros(num_trees).tolist(), num_trees, num_classes, g_table, 0)
    print('Done')

    feature_width = []
    for max_f in feature_max:
        feature_width += [int(np.ceil(math.log(max_f, 2)) + 1)]

    code_width_tree_feature = np.zeros((num_trees,num_features))
    for i in range(num_features):
        for tree in range(num_trees):
            code_width_tree_feature[tree, i] = int(np.ceil(math.log(g_table[tree]['feature ' + str(i)][np.max(list(g_table[tree]['feature ' + str(i)].keys()))]+1,2)+1)) or 1

    table_ternary = {}
    table_ternary['decision'] = g_table['votes to class']

    for tree in range(num_trees):
        table_ternary['tree ' + str(tree)] = g_table[tree]['code to vote']

    for i in range(num_features):
        table_ternary['feature '+str(i)] = {}
        for value in range(feature_max[i]):
            table_ternary['feature ' + str(i)][value] = []
            for tree in range(num_trees):
                table_ternary['feature ' + str(i)][value] += [g_table[tree]["feature "+str(i)][value]]
    table_exact = copy.deepcopy(table_ternary)
    for i in range(num_features):
        if i!=0:
            print('')
        print('Begin transfer: Feature table ' +str (i))
        table_ternary['feature '+str(i)]= Table_to_TCAM(table_ternary['feature '+str(i)],
                                                        feature_width[i])

    # ===================== prepare default vote =========================
    collect_votes = []
    for t in range(num_trees):
        for idx in table_exact['tree '+str(t)]:
            collect_votes += [int(table_exact['tree '+str(t)][idx]['leaf'])]
    default_vote = max(collect_votes, key=collect_votes.count)

    code_table_size = 0
    for t in range(num_trees):
        table_ternary['tree '+str(t)] = {}
        for idx in table_exact['tree '+str(t)]:
            if int(table_exact['tree '+str(t)][idx]['leaf']) != default_vote:
                table_ternary['tree '+str(t)][code_table_size] = table_exact['tree '+str(t)][idx]
                code_table_size += 1
        table_exact['tree '+str(t)] = copy.deepcopy(table_ternary['tree '+str(t)])

    # ===================== prepare default class =========================

    collect_class = []
    for idx in table_exact['decision']:
        collect_class += [table_exact['decision'][idx]['class']]
    default_class = max(collect_class, key=collect_class.count)

    code_table_size = 0
    table_ternary['decision'] = {}
    for idx in table_exact['decision']:
        if table_exact['decision'][idx]['class'] != default_class:
            table_ternary['decision'][code_table_size] = table_exact['decision'][idx]
            code_table_size += 1
    table_exact['decision'] = copy.deepcopy(table_ternary['decision'])

    # =================== convert model timer ===================
    config['timer log']['convert model']['end'] = time.time()
    # ===========================================================

    table_name = 'ternary_table.json'
    json.dump(table_ternary, open(f'tables/{cur_dataset}-{cur_trace}-rf-{num_trees}-{num_depth}-{max_leaf_nodes}-{table_name}', 'w'), indent=4)
    print('\ntable_ternary is generated')
    table_name = 'exact_table.json'
    json.dump(table_exact, open(f'tables/{cur_dataset}-{cur_trace}-rf-{num_trees}-{num_depth}-{max_leaf_nodes}-{table_name}', 'w'), indent=4)
    print('table_exact is generated')

    config['p4 config']                         = {}
    config['p4 config']["model"]                = "rf"
    config['p4 config']["number of features"]   = num_features
    config['p4 config']["number of classes"]    = num_classes
    config['p4 config']["number of trees"]      = num_trees
    config['p4 config']['table name']           = 'ternary_table.json'
    config['p4 config']["decision table size"]  = len(table_ternary['decision'].keys())
    config['p4 config']["code table size"]      = []
    for tree in range(num_trees):
        config['p4 config']["code table size"] += [len(table_ternary['tree '+str(tree)].keys())]
    config['p4 config']["default vote"]     = default_vote
    config['p4 config']["default label"]    = default_class
    config['p4 config']["width of feature"] = feature_width
    config['p4 config']["width of code"]    = code_width_tree_feature
    config['p4 config']["used columns"]     = []
    for i in range(num_features):
        config['p4 config']["used columns"] += [len(table_ternary['feature '+str(i)].keys())]
    config['p4 config']["width of probability"] = 7
    config['p4 config']["width of result"]      =  8
    config['p4 config']["standard headers"]     = [ "ethernet", "Planter", "arp", "ipv4", "tcp", "udp", "vlan_tag" ]
    config['test config']                       = {}
    config['test config']['type of test']       = 'classification'

    json.dump(config,
              open(config['directory config']['work']+"/"+config_path, 'w'),
              indent=4,
              cls=NpEncoder)

    # main()
    return sklearn_y_predict.tolist()

def test_tables(sklearn_test_y, test_X, test_y, cur_dataset, cur_trace,
                config_path=None, threshold=None):
    if config_path:
        print(config_path)
        config = json.load(open(config_path, 'r'))
    else:
        config = json.load(open('conf/planter_config.json', 'r'))

    num_features    = config['data config']['number of features']
    num_trees       = config['model config']['number of trees']
    num_depth       = config['model config']['number of depth']
    max_leaf_nodes  = config['model config']['max number of leaf nodes']
    table_ternary   = json.load(open(f'tables/{cur_dataset}-{cur_trace}-rf-{num_trees}-'
                                     f'{num_depth}-{max_leaf_nodes}-ternary_table.json', 'r'))
    table_exact     = json.load(open(f'tables/{cur_dataset}-{cur_trace}-rf-{num_trees}-'
                                     f'{num_depth}-{max_leaf_nodes}-exact_table.json', 'r'))

    print('Test the exact feature table, exact code and decision table (feel free if the acc to sklearn is slightly lower than 1)')

    same                = 0
    correct             = 0
    error               = 0
    switch_test_y       = []
    switch_test_y_proba = []

    for i in range(np.shape(test_X.values)[0]):
        if i % 10000 == 0:
            print(f'Test: processed {i} packets.')
        vote_list = np.zeros(num_trees).astype(dtype=int).tolist()
        for tree in range(num_trees):
            code_list = np.zeros(num_features)
            ternary_code_list = np.zeros(num_features)
            input_feature_value = test_X.values[i]

            for f in range(num_features):
                match_or_not = False

                # match ternary
                TCAM_table  = table_ternary['feature ' + str(f)]
                keys        = list(TCAM_table.keys())

                for count in keys:
                    if input_feature_value[f] & TCAM_table[count][0] == TCAM_table[count][0] & TCAM_table[count][1]:
                        ternary_code_list[f] = TCAM_table[count][2][tree]
                        match_or_not = True
                        break

                if not match_or_not:
                    print('feature table not matched')
                # match exact
                code_list[f] = table_exact['feature ' + str(f)][str(input_feature_value[f])][tree]
                if not match_or_not:
                    print('feature table not matched')
            if str(code_list)!=str(ternary_code_list):
                print('error in exact to ternary match', code_list,ternary_code_list)
            for key in table_exact["tree " + str(tree)]:

                match_or_not = False
                all_True = True
                for code_f in range(num_features):
                    if not table_exact["tree " + str(tree)][key]['f' + str(code_f) + ' code'] == code_list[code_f]:
                        all_True = False
                        break
                if all_True:
                    vote_list[tree] = int(table_exact["tree " + str(tree)][key]['leaf'])
                    match_or_not = True
                    break
            if not match_or_not:
                vote_list[tree] =  config['p4 config']["default vote"]

        for key in table_exact['decision']:
            match_or_not = False
            all_True = True
            for tree_v in range(num_trees):
                if not table_exact["decision"][key]['t' + str(tree_v) + ' vote'] == vote_list[tree_v]:
                    all_True = False
                    break
            if all_True:
                switch_prediction = table_exact['decision'][key]['class']
                match_or_not = True
                break
        if not match_or_not:
            switch_prediction = config['p4 config']["default label"]

        switch_test_y_proba += [sum(vote_list)/len(vote_list)]
        switch_test_y       += [switch_prediction]

        if switch_prediction == test_y[i]:
            correct += 1

        if switch_prediction == sklearn_test_y[i]:
            same += 1
        else:
            error += 1

    eval_metrics(test_y,
                 switch_test_y,
                 switch_test_y_proba,
                 f'rf-{num_trees}-{num_depth}-{max_leaf_nodes}-switch',
                 cur_dataset,
                 cur_trace)

def resource_prediction(config_path):
    config = json.load(open(config_path, 'r'))

    print('Exact match entries:     ', np.sum(config['p4 config']["code table size"]) \
          + config['p4 config']["decision table size"] )
    print('Ternary match entries:   ', np.sum(config['p4 config']["used columns"]))

if __name__ == '__main__':
    print('there are many dependencies, directly run is not currently supported')
