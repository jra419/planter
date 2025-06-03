# THIS FILE IS PART OF Planter PROJECT
# Planter.py - The core part of the Planter library
#
# THIS PROGRAM IS FREE SOFTWARE TOOL, WHICH MAPS MACHINE LEARNING ALGORITHMS TO DATA PLANE, IS LICENSED UNDER Apache-2.0
# YOU SHOULD HAVE RECEIVED A COPY OF WTFPL LICENSE, IF NOT, PLEASE CONTACT THE FOLLOWING E-MAIL ADDRESSES
#
# Copyright (c) 2020-2021 Changgang Zheng, Mingyuan Zang
# Copyright (c) Computing Infrastructure Lab, Department of Engineering Science, University of Oxford
# E-mail: changgang.zheng@eng.ox.ac.uk or changgangzheng@qq.com
#
# Functions: This file is responsible for training, algorithm mapping, and software testing of the ML model.
#            Please refer to ./Docs/Planter_User_Document.pdf or further information.

from sklearn.neighbors import KNeighborsClassifier
from src.functions.Range_to_TCAM_Top_Down import ten_to_bin
from src.functions.json_encoder import NpEncoder
from eval.eval_metrics import eval_metrics
import numpy as np
import time
import copy
import json


def relative_code_lookup(idx, num_features, feature_num, look_up, label):
    if feature_num ==num_features:
        code = ''

        for f in range(num_features):
            code += str(int(idx[f]))

        look_up[code] = label
        label        += 1

        return look_up, label
    else:
        for r in [0,1]:
            idx[feature_num]    = r
            feature_num        += 1
            look_up,label       = relative_code_lookup(idx,num_features,
                                                       feature_num, look_up, label)
            feature_num        -= 1

    return look_up, label


def get_codes(lookup, x, num_features, num_depth, depth_num, border_max, border_min):
    code        = ''
    need_split  = True

    while need_split:
        if num_depth == depth_num:
            break

        center  = np.zeros(num_features)
        con     = ''

        for f in range(num_features):
            center[f] = (copy.deepcopy(border_max[f]) + copy.deepcopy(border_min[f])) / 2

        for f in range(num_features):
            if x[f]>= center[f]:
                con            += '1'
                border_min[f]   = copy.deepcopy(center[f])
            else:
                con            += '0'
                border_max[f]   = copy.deepcopy(center[f])

        code        += ten_to_bin(lookup[con], num_features)
        depth_num   += 1

    return code

def get_boarder_list(border_dict, num_features, feature_num, num_depth, border_max,
                     border_min, value_list, idx):
    if num_features == feature_num:
        border_dict[idx] = copy.deepcopy(value_list)
        idx             += 1

        return border_dict, idx
    else:
        for i in [0,1]:
            if i == 0:
                value_list[feature_num] = border_min[feature_num]
            else:
                value_list[feature_num] = border_max[feature_num]

            feature_num     += 1
            border_dict, idx = get_boarder_list(border_dict, num_features, feature_num, num_depth, 
                                                border_max, border_min, value_list, idx)
            feature_num     -= 1

    return border_dict, idx


def check_if_not_finish(center,width, division, num_features,num_depth, num_classes, knn_clf):
    not_finish      = False
    border_max_test = np.zeros(num_features)
    border_min_test = np.zeros(num_features)

    for f in range(num_features):
        border_min_test[f] = center[f] - width[f]
        border_max_test[f] = center[f] + width[f]

    border_list     = {}
    border_list, _  = get_boarder_list(border_list, num_features, 0, num_depth, border_max_test, 
                                       border_min_test, np.zeros(num_features), 0)
    x_border        = []

    for idx in border_list:
        x_border += [list(border_list[idx])]

    y_border = knn_clf.predict(x_border)

    if len(np.unique(y_border)) !=1:
        not_finish  = True
        cla         = 404
    else:
        cla = np.unique(y_border)[0]

    return not_finish, cla

def clustream(table, idx, code, lookup, num_features, num_classes, num_depth, depth_num, center,
              width, border_max, border_min, knn_clf, division, is_return):
    cla = 0
    if not is_return:
    # is_return == False
        for f in range(num_features):
            center[f] = (copy.deepcopy(border_max[f]) + copy.deepcopy(border_min[f])) / 2
            width[f] = (copy.deepcopy(border_max[f]) - copy.deepcopy(border_min[f])) / 2


        not_finish = True
        if depth_num ==0:
            not_finish = True
        elif depth_num < num_depth:
            not_finish, cla = check_if_not_finish(copy.deepcopy(center), copy.deepcopy(width),
                                                division, num_features, num_depth, num_classes,
                                                knn_clf)
    else:
        not_finish = False
        # cla = 404
        _, cla = check_if_not_finish(copy.deepcopy(center), copy.deepcopy(width), division,
                                     num_features, num_depth, num_classes, knn_clf)

    if not_finish:
        for division in lookup:
            new_boarder_max = np.zeros(num_features)
            new_boarder_mim = np.zeros(num_features)
            for f in range(num_features):
                if division[f] == '0':
                    new_boarder_mim[f] = copy.deepcopy(center[f]) - copy.deepcopy(width[f])
                    new_boarder_max[f] = copy.deepcopy(center[f])
                else:
                    new_boarder_max[f] = copy.deepcopy(center[f]) + copy.deepcopy(width[f])
                    new_boarder_mim[f] = copy.deepcopy(center[f])
            depth_num   += 1
            code        += ten_to_bin(lookup[division], num_features)

            table, idx, is_return = clustream(table,
                                              idx,
                                              copy.deepcopy(code),
                                              lookup,
                                              num_features, num_classes, num_depth, depth_num,
                                              copy.deepcopy(center),
                                              copy.deepcopy(width),
                                              copy.deepcopy(new_boarder_max),
                                              copy.deepcopy(new_boarder_mim), knn_clf,
                                              division, is_return)
            # table, idx, is_return = clustream(table, idx, copy.deepcopy(code), lookup,
            #                                   num_features, num_classes, num_depth, copy.deepcopy(depth_num), copy.deepcopy(center),
            #                                   copy.deepcopy(width), 
            #                                   copy.deepcopy(new_boarder_max),
            #                                   copy.deepcopy(new_boarder_mim), knn_clf,
            #                                   division, is_return)
            depth_num  -= 1
            code        = code[:-num_features]
    else:
        mask        = (depth_num)*(num_features*'1')+(num_depth-depth_num)*(num_features*'0')
        value       = code+(num_depth-depth_num)*(num_features*'0')
        table[idx]  = [int(mask,2), int(value,2), cla]
        idx        += 1

        if is_return:
            return table, idx, is_return

    if is_return:
        return table, idx, is_return

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
    num_neighbours  = config['model config']['number of neighbours']
    num_classes     = config['model config']['number of classes']
    num_depth       = config['model config']['depth of quadtree']

    print("TEST 1")

    feature_names = []
    for i, f in enumerate(used_features):
        train_X.rename(columns={f: "f" + str(i)}, inplace=True)
        test_X.rename(columns={f: "f" + str(i)}, inplace=True)
        feature_names += ["f" + str(i)]

    feature_max = []
    for i in feature_names:
        t_t = [test_X[[i]].max()[0], train_X[[i]].max()[0]]
        feature_max += [np.max(t_t) + 1]

    feature_min = []
    for i in feature_names:
        t_t = [test_X[[i]].min()[0], train_X[[i]].min()[0]]
        feature_min += [np.min(t_t) ]


    # =================== train model timer ===================
    config['timer log']['train model']          = {}
    config['timer log']['train model']['start'] = time.time()
    # =================== train model timer ===================

    # knn fit
    print("TEST 2")
    knn_clf = KNeighborsClassifier(n_neighbors=num_neighbours, algorithm='kd_tree')
    print("TEST 3")
    knn_clf.fit(train_X, train_y)

    print("TEST 4")
    sklearn_y_predict       = knn_clf.predict(test_X)
    print("TEST 5")
    sklearn_y_predict_proba = knn_clf.predict_proba(test_X)[:,1]
    print("TEST 6")

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
    config['timer log']['convert model']            = {}
    config['timer log']['convert model']['start']   = time.time()
    # =================== convert model timer ===================

    print('Generating Ternary Tables... ')

    lookup      = {}
    lookup, _   = relative_code_lookup(np.zeros(num_features), num_features, 0, lookup, 0)

    Ternary_Table       = {}
    Ternary_Table, _, _ = clustream(Ternary_Table, 0, '', lookup, num_features, num_classes, 
                                    num_depth, 0, np.zeros(num_features), np.zeros(num_features), 
                                    feature_max, feature_min, knn_clf, '', True)

    # =================== convert model timer ===================
    config['timer log']['convert model']['end'] = time.time()
    # =================== convert model timer ===================

    json.dump(Ternary_Table, open(f'eval/tables/{cur_dataset}/{cur_model}/{cur_trace}-{cur_model}-'
                                f'{model_size}-ternary_table.json', 'w'), indent=4, cls=NpEncoder)

    # ========================== prepare the test data =====================================
    for i in range(np.shape(test_X.values)[0]):
        # distance            = np.zeros(num_classes).tolist()
        input_feature_value = test_X.values[i]
        code                = get_codes(lookup, input_feature_value, num_features, num_depth, 0, 
                                        copy.deepcopy(feature_max), copy.deepcopy(feature_min))
        test_X.values[i][0] = int(code, 2)
    # =======================================================================================

    config['p4 config']                         = {}
    config['p4 config']["model"]                = "knn"
    config['p4 config']["number of features"]   = num_features
    config['p4 config']["number of classes"]    =  num_classes
    config['p4 config']["table length"]         = len(Ternary_Table.keys())
    config['p4 config']['table name']           = f'{cur_trace}-{cur_model}-{model_size}-ternary_table.json'
    config['model config']['lookup']            = lookup
    config['model config']['feature max']       = feature_max
    config['model config']['feature min']       = feature_min
    config['test config']                       = {}
    config['test config']['type of test']       = 'classification'

    json.dump(config,
              open(config['directory config']['work'] + '/' + config_path, 'w'),
              indent=4,
              cls=NpEncoder)

    return sklearn_y_predict.tolist()

def test_tables(sklearn_test_y, test_X, test_y, cur_dataset, cur_trace,
                config_path=None, threshold=None):
    if config_path:
        print(config_path)
        config = json.load(open(config_path, 'r'))
    else:
        config = json.load(open('conf/planter_config.json', 'r'))

    cur_model       = config['model config']['model']
    model_size      = config['model config']['model size']

    Ternary_Table = json.load(open(f'eval/tables/{cur_dataset}/{cur_model}/{cur_trace}-{cur_model}-{model_size}-ternary_table.json', 'r'))

    same            = 0
    correct         = 0
    error           = 0
    switch_test_y   = []

    for i in range(np.shape(test_X.values)[0]):
        code            = test_X.values[i][0]
        match_or_not    = False
        keys            = list(Ternary_Table.keys())
        print(len(keys))

        for count in keys:
            if code & Ternary_Table[count][0] == Ternary_Table[count][0] & Ternary_Table[count][1]:
                switch_prediction   = Ternary_Table[count][2]
                match_or_not        = True
                break

        if not match_or_not:
            print('feature table not matched')

        switch_test_y += [switch_prediction]

        if switch_prediction == test_y[i]:
            correct += 1

        if switch_prediction == sklearn_test_y[i]:
            same += 1
        else:
            error += 1

        # if i % 10 == 0 and i != 0:
        #     print(
        #         '\rswitch_prediction: {}, test_y: {}, with acc: {:.3}, with acc to sklearn: {:.4}, with error: {:.3}, M/A format macro f1: {:.3}, macro f1: {:.3}'.format(
        #             switch_prediction, test_y[i], correct / (i + 1), same / (i + 1), error / (i + 1),
        #             accuracy_score(switch_test_y[:i], test_y[:i]), accuracy_score(sklearn_test_y[:i], test_y[:i])),
        #         end="")

    # print('\nThe accuracy of the match action format of Kmeans is', correct / np.shape(test_X.values)[0])
    # result = classification_report(switch_test_y, test_y, digits=4)
    # print('\n', result)

def resource_prediction(config_path):
    config = json.load(open(config_path, 'r'))

    print('Ternary match entries: ',np.sum(config['p4 config']["table length"]) )
