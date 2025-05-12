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


from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from src.functions.normalization import Single_MaxMinNormalization
from eval.eval_metrics import eval_metrics
from src.functions.json_encoder import NpEncoder
import copy
import math
import numpy as np
import time
import os
import sys
import json

global_model_parameters = 0
global_separate_table   = 0

def ten_to_bin(num,count):
    num = int(num)
    num = bin(num).lstrip('0b')

    if len(num) != count:
        cont = count - len(num)
        num = cont * '0' + num
    return num


def calculate_prob(input,feature_No, class_No, model_parmeters):#i is class
    part_1 = 1 / np.sqrt(2 * np.pi * model_parmeters['c'+str(class_No)]['f'+str(feature_No)]['std'] ** 2)
    part_2_u =   (input - model_parmeters['c'+str(class_No)]['f'+str(feature_No)]['mean']) ** 2
    part_2_l =  2 * (model_parmeters['c'+str(class_No)]['f'+str(feature_No)]['std'] ** 2)
    return part_1*np.exp(-part_2_u/part_2_l)

def run_model(train_X, train_y, test_X, test_y, used_features, cur_dataset,
              cur_trace, config_path=None):
    if config_path:
        print(f'Config: {config_path}')
        config = json.load(open(config_path, 'r'))
    else:
        config = json.load(open('conf/planter_config.json', 'r'))

    cur_model       = config['model config']['model']
    model_size      = config['model config']['model size']
    num_bits        = config['model config']['number of bits']
    num_features    = config['data config']['number of features']
    num_classes     = config['model config']['number of classes']

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
    # =================== train model timer ===================

    # fit
    clf = GaussianNB()
    clf.fit(train_X, train_y)
    sklearn_y_predict       = clf.predict(test_X)
    sklearn_y_predict_proba = clf.predict_proba(test_X)[:,1]

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

    model_parmeters={}
    for c in range(num_classes):
        model_parmeters['c'+str(c)] = {}
        for f in range(num_features):
            model_parmeters['c' + str(c)]["f"+str(f)] = {}
            model_parmeters['c' + str(c)]["f"+str(f)]['std'] = np.sqrt(clf.var_[c,f])
            model_parmeters['c' + str(c)]["f"+str(f)]['mean'] = clf.theta_[c,f]

    value_info = {}
    value_info["max"] = 0
    value_info["min"] = 0
    for f in range(num_features):
        value_info["f" + str(f)] = {}
        value_info["f"+str(f)]["max"] =  0
        value_info["f"+str(f)]["min"] =  0

    Bayes_separate_table = {}
    for f in range(num_features):
        Bayes_separate_table['feature '+str(f)] = {}
        for inputs in range(0,feature_max[f]+1):
            Bayes_separate_table['feature '+str(f)][inputs]={}
            for c in range(num_classes):
                if calculate_prob(inputs,f,c,model_parmeters)==0:
                    value = value_info["min"]
                else:
                    value = math.log(calculate_prob(inputs,f,c,model_parmeters),2)
                Bayes_separate_table['feature '+str(f)][inputs]["class "+str(c)] = value
                if value > value_info["max"]:
                    value_info["max"] = value
                if value < value_info["min"]:
                    value_info["min"] = value

    Bayes_separate_table["class prob"] = {}
    for c in range(num_classes):
        value = clf.class_prior_[c]
        Bayes_separate_table["class prob"]['class '+str(c)]= math.log(value,2)
        if value > value_info["max"]:
            value_info["max"] = value
        if value < value_info["min"]:
            value_info["min"] = value

    scale = (2**num_bits)/(num_features+1)
    Exact_Table = {}
    Exact_Table['class prob'] = {}
    for c in range(num_classes):
        min_x = value_info["min"]
        max_x = value_info["max"]
        x = copy.deepcopy(Bayes_separate_table['class prob']['class '+str(c)])
        value = Single_MaxMinNormalization(x, min_x, max_x)
        Exact_Table['class prob']['class '+str(c)] = int(np.round(value*scale))


    for f in range(num_features):
        Exact_Table['feature '+str(f)] = {}
        for inputs in range(0,feature_max[f]+1):
            Exact_Table['feature ' + str(f)][inputs] = {}
            for c in range(num_classes):
                min_x = value_info["min"]
                max_x = value_info["max"]
                x = Bayes_separate_table['feature '+str(f)][inputs]["class "+str(c)]
                value = Single_MaxMinNormalization(x, min_x, max_x)
                Exact_Table['feature '+str(f)][inputs]["class "+str(c)] = int(np.round(value*scale))

    global global_model_parameters, global_separate_table
    global_model_parameters = model_parmeters
    global_separate_table = Bayes_separate_table

    # =================== convert model timer ===================
    config['timer log']['convert model']['end'] = time.time()
    # =================== convert model timer ===================

    json.dump(Exact_Table, open(f'eval/tables/{cur_dataset}/{cur_model}/{cur_trace}-{cur_model}-'
                                f'{model_size}-exact_table.json', 'w'), indent=4)

    feature_tbl_len = []
    for f in range(num_features):
        feature_tbl_len += [len(Exact_Table['feature ' + str(f)].keys())]

    config['p4 config']                         = {}
    config['p4 config']["model"]                = "bayes"
    config['p4 config']["number of features"]   = num_features
    config['p4 config']["number of classes"]    = num_classes
    config['p4 config']["action data bits"]     = num_bits
    config['p4 config']['table name']           = f'{cur_trace}-{cur_model}-{model_size}-exact_table.json'
    config['p4 config']["feature tbl len"]      = feature_tbl_len
    config['test config']                       = {}
    config['test config']['type of test']       = 'classification'


    json.dump(config,
              open(config['directory config']['work']+"/"+config_path, 'w'),
              indent=4,
              cls=NpEncoder)

    return sklearn_y_predict.tolist()

def predict_probabilities(input_features, model_parameters, Bayes_separate_table, num_classes):
    log_probs = {}

    # Calculate log probabilities for each class
    for c in range(num_classes):
        log_prob = Bayes_separate_table["class prob"]['class ' + str(c)]

        for f, feature_value in enumerate(input_features):
            if feature_value in Bayes_separate_table['feature ' + str(f)]:
                log_prob += Bayes_separate_table['feature ' + str(f)][feature_value]["class " + str(c)]
            else:
                # Handle unseen feature values (could be set to a small value or ignored)
                log_prob += -np.inf  # Log(0) is -inf, indicating zero probability

        log_probs[c] = log_prob

    # Convert log probabilities to normal probabilities
    max_log_prob = max(log_probs.values())
    exp_probs = {c: math.exp(log_prob - max_log_prob) for c, log_prob in log_probs.items()}

    # Normalize probabilities
    total_prob = sum(exp_probs.values())
    probabilities = {c: prob / total_prob for c, prob in exp_probs.items()}

    return probabilities

def test_tables(sklearn_test_y, test_X, test_y, cur_dataset, cur_trace,
                config_path=None, threshold=None):
    if config_path:
        print(config_path)
        config = json.load(open(config_path, 'r'))
    else:
        config = json.load(open('conf/planter_config.json', 'r'))

    num_features    = config['data config']['number of features']
    num_classes     = config['model config']['number of classes']
    cur_model       = config['model config']['model']
    model_size      = config['model config']['model size']

    Exact_Table     = json.load(open(f'eval/tables/{cur_dataset}/{cur_model}/{cur_trace}-'
                                     f'{cur_model}-{model_size}-exact_table.json', 'r'))

    same    = 0
    correct = 0
    error   = 0

    switch_test_y       = []
    switch_test_y_proba = []

    for i in range(np.shape(test_X.values)[0]):
        input_feature_value = test_X.values[i]
        class_prob = np.zeros(num_classes).tolist()

        for c in range(num_classes):
            class_prob[c] = Exact_Table['class prob']['class '+str(c)]

        for f in range(num_features):
            for c in range(num_classes):
                class_prob[c] += Exact_Table['feature '+str(f)][str(input_feature_value[f])]['class '+str(c)]

        switch_prediction = class_prob.index(np.max(class_prob))
        switch_test_y += [switch_prediction]

        global global_model_parameters, global_separate_table

        probabilities = predict_probabilities(test_X.values[i], global_model_parameters, global_separate_table, num_classes)

        switch_test_y_proba += [probabilities[switch_prediction]]

        if switch_prediction == test_y[i]:
            correct += 1

        if switch_prediction == sklearn_test_y[i]:
            same += 1
        else:
            error += 1

    eval_metrics(test_y,
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
