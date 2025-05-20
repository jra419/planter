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

from sklearn.cluster import KMeans
from src.functions.json_encoder import NpEncoder
from eval.eval_metrics import eval_metrics_kmeans
import pandas as pd
import numpy as np
import time
import copy
import json


def run_model(train_X, train_y, test_X, test_y, used_features, cur_dataset,
              cur_trace, config_path=None):
    cur_X = pd.concat([train_X, test_X])
    cur_y = np.concatenate((train_y, test_y), axis=0)

    last_n = cur_dataset[-3:]
    if last_n == '-ad':
        cur_dataset = cur_dataset [:-3]

    if config_path:
        print(f'Config: {config_path}')
        config = json.load(open(config_path, 'r'))
    else:
        config = json.load(open('conf/planter_config.json', 'r'))

    num_features    = config['data config']['number of features']
    cur_model       = config['model config']['model']
    model_size      = config['model config']['model size']
    random_state    = config['model config']['random state']
    num_bits        = config['model config']['number of bits']
    num_classes     = config['model config']['number of classes']

    feature_names = []
    for i, f in enumerate(used_features):
        # train_X.rename(columns={f: "f" + str(i)}, inplace=True)
        # test_X.rename(columns={f: "f" + str(i)}, inplace=True)
        cur_X.rename(columns={f: "f" + str(i)}, inplace=True)
        feature_names += ["f" + str(i)]

    feature_max = []
    for i in feature_names:
        # t_t             = [test_X[[i]].max()[0], train_X[[i]].max()[0]]
        # feature_max    += [np.max(t_t) + 1]
        feature_max    += [cur_X[[i]].max()[0] + 1]

    feature_min = []
    for i in feature_names:
        # t_t             = [test_X[[i]].min()[0], train_X[[i]].min()[0]]
        # feature_min    += [np.min(t_t) ]
        feature_min    += [np.min(cur_X[[i]].min()[0])]

    # =================== train model timer ===================
    config['timer log']['train model'] = {}
    config['timer log']['train model']['start'] = time.time()
    # =================== train model timer ===================

    kmeans = KMeans(n_clusters=num_classes,
                    random_state=random_state,
                    n_init=random_state).fit(cur_X)

    sklearn_y_predict = kmeans.predict(cur_X)

    eval_metrics_kmeans(cur_y, sklearn_y_predict, cur_dataset, cur_trace, cur_model,
                        model_size, 'sklearn')

    # =================== train model timer ===================
    config['timer log']['train model']['end'] = time.time()
    # =================== train model timer ===================

    # =================== convert model timer ===================
    config['timer log']['convert model'] = {}
    config['timer log']['convert model']['start'] = time.time()
    # =================== convert model timer ===================

    centre = kmeans.cluster_centers_

    # record the model
    outputfile  = 'src/tmp/km.txt'
    centers     = {}
    model       = open(outputfile,"w+")

    for c in range(len(centre)):
        model.write("centre point for class "+str(c)+" : \n")
        centers["c"+str(c)] = {}
        model.write("(")

        for f in range(num_features):
            centers["c"+str(c)]['f'+str(f)] = centre[c][f]
            if f+1 >= num_features:
                model.write('f' + str(f) + ': ' + str(centre[c][f]) + ")")
            else:
                model.write( 'f'+str(f)+': '+str(centre[c][f]) + ", " )
        model.write(";\n")

    model.close()

    Tables              = {}
    value_info          = {}
    value_info["max"]   = 0

    for f in range(num_features):
        Tables['feature ' + str(f)] = {}

        x_m     = np.mean(cur_X[feature_names[f]])
        x_std   = np.std(cur_X[feature_names[f]])

        for input_value in range(feature_min[f], feature_max[f]):
            Tables['feature '+str(f)][input_value] = {}

            for c in range(num_classes):
                value = (centers["c"+str(c)]['f'+str(f)] - input_value) ** 2
                Tables['feature ' + str(f)][input_value]["c" + str(c)] = value
                if value > value_info["max"]:
                    value_info["max"] = value

    if num_bits != 0:
        scale = (2**num_bits)/ (value_info["max"]*num_features)

    Exact_Table = {}

    for f in range(num_features):
        Exact_Table['feature ' + str(f)] = {}

        for input_value in range(feature_min[f], feature_max[f]):
            Exact_Table['feature ' + str(f)][input_value] = {}

            for c in range(num_classes):
                value = copy.deepcopy(Tables['feature ' + str(f)][input_value]["c" + str(c)])
                if num_bits != 0:
                    value = int(np.floor(value*scale))
                Exact_Table['feature ' + str(f)][input_value]["c" + str(c)] = value

    # =================== convert model timer ===================
    config['timer log']['convert model']['end'] = time.time()
    # =================== convert model timer ===================

    json.dump(Exact_Table, open(f'eval/tables/{cur_dataset}/{cur_model}/{cur_trace}-{cur_model}-'
                                f'{model_size}-exact_table.json', 'w'), indent=4)

    feature_tbl_len = []
    for f in range(num_features):
        feature_tbl_len += [len(Exact_Table['feature ' + str(f)].keys())]

    config['p4 config'] = {}

    config['p4 config']["model"]                = "km"
    config['p4 config']["number of features"]   = num_features
    config['p4 config']["number of classes"]    =  num_classes
    config['p4 config']["action data bits"]     = num_bits
    config['p4 config']['table name']           = f'{cur_trace}-{cur_model}-{model_size}-ternary_table.json'
    config['p4 config']["feature tbl len"]      = feature_tbl_len
    config['test config']                       = {}
    config['test config']['type of test']       = 'classification'

    json.dump(config,
              open(config['directory config']['work']+"/"+config_path, 'w'),
              indent=4,
              cls=NpEncoder)

    return sklearn_y_predict.tolist()

def test_tables(sklearn_y, train_X, train_y, test_X, test_y, cur_dataset, cur_trace,
                config_path=None, threshold=None):
    cur_X = pd.concat([train_X, test_X])
    cur_y = np.concatenate((train_y, test_y), axis=0)

    last_n = cur_dataset[-3:]
    if last_n == '-ad':
        cur_dataset = cur_dataset [:-3]

    if config_path:
        print(config_path)
        config = json.load(open(config_path, 'r'))
    else:
        config = json.load(open('conf/planter_config.json', 'r'))

    num_features    = config['data config']['number of features']
    cur_model       = config['model config']['model']
    model_size      = config['model config']['model size']
    num_classes     = config['model config']['number of classes']

    Exact_Table     = json.load(open(f'eval/tables/{cur_dataset}/{cur_model}/{cur_trace}-'
                                     f'{cur_model}-{model_size}-exact_table.json', 'r'))

    switch_y = []

    for i in range(np.shape(cur_X.values)[0]):
        distance = np.zeros(num_classes).tolist()
        input_feature_value = cur_X.values[i]

        for f in range (num_features):
            for c in range(num_classes):
                distance[c] += Exact_Table['feature ' + str(f)][str(input_feature_value[f])]["c"+str(c)]

        switch_prediction   = distance.index(np.min(distance))
        switch_y           += [switch_prediction]

    eval_metrics_kmeans(cur_y, switch_y, cur_dataset, cur_trace, cur_model,
                        model_size, 'sklearn')

def resource_prediction(config_path):
    config = json.load(open(config_path, 'r'))

    print('Exact match entries:     ', np.sum(config['p4 config']["code table size"]) \
          + config['p4 config']["decision table size"] )
