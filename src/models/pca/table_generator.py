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

import numpy as np
from sklearn.decomposition import PCA
import json
import copy
from src.functions.json_encoder import NpEncoder
from src.functions.normalization import *
import pandas as pd
import time

def run_model(train_X, train_y, test_X, test_y, used_features, cur_dataset,
              cur_trace, config_path=None):
    cur_X = pd.concat([train_X, test_X])

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
    num_bits        = config['model config']['number of bits']
    num_components  = config['model config']['num components']
    num_classes     = config['model config']['number of classes']

    feature_names = []
    for i, f in enumerate(used_features):
        train_X.rename(columns={f: "f" + str(i)}, inplace=True)
        test_X.rename(columns={f: "f" + str(i)}, inplace=True)
        feature_names += ["f" + str(i)]

    feature_max = []
    for i in feature_names:
        t_t = [test_X[[i]].max()[0], train_X[[i]].max()[0]]
        feature_max += [max(t_t)+1]

    # =================== train model timer ===================
    config['timer log']['train model'] = {}
    config['timer log']['train model']['start'] = time.time()
    # =================== train model timer ===================

    pca             = PCA(n_components=num_components)
    sklearn_X_new   = pca.fit_transform(cur_X)

    # =================== train model timer ===================
    config['timer log']['train model']['end'] = time.time()
    # =================== train model timer ===================

    # =================== convert model timer ===================
    config['timer log']['convert model'] = {}
    config['timer log']['convert model']['start'] = time.time()
    # =================== convert model timer ===================

    model_info                  = {}
    model_info['means']         = pca.mean_
    model_info['components']    = pca.components_.T

    value_info          = {}
    value_info["max"]   = 0
    value_info["min"]   = 0

    for ax in range(num_components):
        value_info["ax "+str(ax)]           = {}
        value_info["ax " + str(ax)]["max"]  = 0
        value_info["ax " + str(ax)]["min"]  = 0

    PCA_Table = {}

    for f in range(num_features):
        PCA_Table['feature '+str(f)] = {}

        for input_value in range(feature_max[f]):
            PCA_Table['feature ' + str(f)][input_value] = {}
            value = input_value - model_info['means'][f]

            for ax in range(num_components):
                middle_value = copy.deepcopy(value*model_info['components'][f,ax])
                PCA_Table['feature ' + str(f)][input_value]['ax'+str(ax)] = middle_value

                if middle_value > value_info["ax " + str(ax)]["max"]:
                    value_info["ax " + str(ax)]["max"] = middle_value
                if middle_value < value_info["ax " + str(ax)]["min"]:
                    value_info["ax " + str(ax)]["min"] = middle_value
                if middle_value > value_info["max"]:
                    value_info["max"] = middle_value
                if middle_value < value_info["min"]:
                    value_info["min"] = middle_value

    if num_bits != 0:
        scale = (2**num_bits)/((value_info["max"]-value_info["min"])*(num_features))

    Exact_Table = {}

    for f in range(num_features):
        Exact_Table['feature ' + str(f)] = {}

        for input_value in range(feature_max[f]):
            Exact_Table['feature ' + str(f)][input_value] = {}

            for ax in range(num_components):
                middle_value = copy.deepcopy(PCA_Table['feature ' + str(f)][input_value]['ax' + str(ax)])
                if num_bits != 0:
                    middle_value = int(np.floor((middle_value - value_info["min"])*scale))
                Exact_Table['feature ' + str(f)][input_value]['ax' + str(ax)] = middle_value

    # =================== convert model timer ===================
    config['timer log']['convert model']['end'] = time.time()
    # =================== convert model timer ===================

    json.dump(Exact_Table, open(f'eval/tables/{cur_dataset}/{cur_model}/{cur_trace}-{cur_model}-'
                                f'{model_size}-exact_table.json', 'w'), indent=4)

    feature_tbl_len = []

    for f in range(num_features):
        feature_tbl_len += [len(Exact_Table['feature ' + str(f)].keys())]

    config['p4 config']                         = {}
    config['p4 config']["model"]                = "PCA"
    config['p4 config']["number of features"]   = num_features
    config['p4 config']["number of classes"]    = num_classes
    config['p4 config']["action data bits"]     = num_bits
    config['p4 config']['table name']           = (
        f'{cur_trace}-{cur_model}-{model_size}-exact_table.json')
    config['p4 config']["feature tbl len"]      = feature_tbl_len
    config['p4 config']["num components"]       = num_components
    config['test config']                       = {}
    config['test config']['type of test']       = 'dimension_reduction'

    json.dump(config,
              open(config['directory config']['work']+"/"+config_path, 'w'),
              indent=4,
              cls=NpEncoder)

    X_new = copy.deepcopy(sklearn_X_new)

    for ax in range(num_components):
        X_new[:, ax] = sklearn_X_new[:, ax] - num_features*(value_info["min"])

    # for ax in range(num_components):
    #     corr, _ = pearsonr(X_new[:, ax],sklearn_X_new[:, ax])
    #     print('Pearsons correlation for axis '+str(ax)+' is: %.3f' % corr)

    return X_new

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
    cur_dataset     = config['data config']['dataset']
    cur_trace       = config['data config']['cur_trace']
    cur_model       = config['model config']['model']
    model_size      = config['model config']['model size']
    num_components  = config['model config']['num components']

    Exact_Table     = json.load(open(f'eval/tables/{cur_dataset}/{cur_model}/{cur_trace}-'
                                     f'{cur_model}-{model_size}-exact_table.json', 'r'))

    print("Test the generated table")

    switch_y = copy.deepcopy(sklearn_y)


    for i in range(np.shape(cur_X.values)[0]):
        input_feature_value = cur_X.values[i]
        for ax in range(num_components):
            switch_y[i][ax] = 0
        for f in range(num_features):
            ax_middle = Exact_Table["feature "+str(f)][str(input_feature_value[f])]
            for ax in range(num_components):
                switch_y[i][ax] += ax_middle["ax"+str(ax)]

    cur_pca_feats = pd.DataFrame(switch_y, columns=['pca_0', 'pca_1'])
    cur_pca_feats.to_csv(f'eval/feats/{cur_dataset}/{cur_model}/{cur_trace}-'
                        f'{cur_model}-{model_size}-feats.csv', index=False)

    # for ax in range(num_components):
        # corr, _ = pearsonr(sklearn_y[:, ax],switch_y[:, ax])
        # print('Pearsons correlation of M/A PCA and output of Sklearn for axis '+str(ax)+' is: %.4f' % corr)

def resource_prediction(config_path):
    config = json.load(open(config_path, 'r'))

    print('Exact match entries:     ', np.sum(config['p4 config']["code table size"]) \
          + config['p4 config']["decision table size"] )
