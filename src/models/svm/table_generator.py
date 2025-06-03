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


from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from datetime import datetime
from src.functions.json_encoder import NpEncoder
from src.functions.normalization import *
from eval.eval_metrics import eval_metrics
from pathos.multiprocessing import ProcessingPool
import numpy as np
import time
import json
import os
import math

def ten_to_bin(num,count):
    num = num.astype(int)
    num = bin(num).lstrip('0b')

    if len(num) != count:
        cont    = count - len(num)
        num     = cont * '0' + num
    return num

def votes_to_class(class_num, vote_list, num_votes, num_classes, g_table, num):
    if class_num  == num_classes:
        if np.sum(vote_list) == num_votes:
            g_table['decision'][num] = {}

            for c in range(num_classes):
                g_table['decision'][num]['c' + str(c) + ' vote'] = vote_list[c]

            g_table['decision'][num]['class'] = vote_list.index(np.max(vote_list))
            num += 1

        return g_table, num
    else:
        for v in range(num_votes+1):
            vote_list[class_num] = v
            class_num           += 1
            g_table, num        = votes_to_class(class_num, vote_list, num_votes, num_classes, 
                                                 g_table, num)
            class_num           -= 1

    return g_table, num

def run_model(train_X, train_y, test_X, test_y, used_features, cur_dataset,
              cur_trace, config_path=None):
    if config_path:
        print(f'Config: {config_path}')
        config = json.load(open(config_path, 'r'))
    else:
        config = json.load(open('conf/planter_config.json', 'r'))

    num_features    = config['data config']['number of features']
    num_bits        = config['model config']['number of bits']
    cur_model       = config['model config']['model']
    model_size      = config['model config']['model size']
    num_classes     = config['model config']['number of classes']

    num_hps         = num_classes * (num_classes - 1) / 2

    feature_names = []
    for i, f in enumerate(used_features):
        train_X.rename(columns={f: "f"+str(i)}, inplace=True)
        test_X.rename(columns={f: "f" + str(i)}, inplace=True)
        feature_names+=["f"+str(i)]

    feature_max = []
    for i in feature_names:
        t_t = [test_X[[i]].max()[0], train_X[[i]].max()[0]]
        feature_max += [max(t_t)]

    # =================== train model timer ===================
    config['timer log']['train model'] = {}
    config['timer log']['train model']['start'] = time.time()
    # =================== train model timer ===================

    # SVM instance
    # SVM = SVC(kernel = 'linear', probability=True)
    # SVM.fit(train_X, train_y)
    # sklearn_y_predict           = SVM.predict(test_X)
    # sklearn_y_predict_proba     = SVM.predict_proba(test_X)[:,1]

    svm = LinearSVC()
    clf = CalibratedClassifierCV(svm)
    clf.fit(train_X, train_y)
    sklearn_y_predict           = clf.predict(test_X)
    # # y_proba = clf.predict_proba(X_test)
    sklearn_y_predict_proba     = clf.predict_proba(test_X)[:,1]

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

    # print(clf.calibrated_classifiers_)

    coe = 0
    int = 0
    for i in clf.calibrated_classifiers_:
        coe = coe + i.estimator.coef_
        int = int + i.estimator.intercept_
    coe = coe/len(clf.calibrated_classifiers_)
    int = int/len(clf.calibrated_classifiers_)

    # coe = clf.base_estimator_.coef_
    # int = clf.base_estimator_.intercept_

    outputfile  = 'src/tmp/svm.txt'
    model       = open(outputfile,"w+")

    for i in range(len(coe)):
        model.write("hyperplane"+str(i)+" = ")

        for f in range(num_features):
            model.write(str(coe[i][f]) + "x"+str(f+1)+" + ")

        model.write(str(int[i]))
        model.write(";\n")

    model.close()

    # Table without fitting to switch
    SVM_separate_table = {}

    value_info = {}
    hps_test = math.floor(num_hps)
    # for hp in range(num_hps):
    for hp in range(hps_test):
        SVM_separate_table["bias hp" + str(hp)] = int[hp]
        value_info["hp " + str(hp)]             = {}
        value_info["hp " + str(hp)]["max"]      = int[hp]
        value_info["hp " + str(hp)]["min"]      = int[hp]

    for i,fn in enumerate(feature_names):

        SVM_separate_table[fn] = {}

        for feature in range(feature_max[i]+1):
            SVM_separate_table[fn][feature] = {}

            # for hp in range(num_hps):
            for hp in range(hps_test):
                middle_value = coe[hp][i] * feature
                SVM_separate_table[fn][feature]["hp " + str(hp)] = middle_value

                if middle_value > value_info["hp " + str(hp)]["max"]:
                    value_info["hp " + str(hp)]["max"] = middle_value
                if middle_value < value_info["hp " + str(hp)]["min"]:
                    value_info["hp " + str(hp)]["min"] = middle_value

    # for hp in range(num_hps):
    for hp in range(hps_test):
        SVM_separate_table['threshold hp'+str(hp)] = 0

    # Table fit to switch
    scale = np.floor((2**num_bits)/ (value_info["hp " + str(hp)]["max"] - value_info["hp " + str(hp)]["min"])/num_features)
    Exact_Table = {}

    print("Generating decision table...", end="")
    Exact_Table['decision'] = {}
    # Exact_Table, _ = votes_to_class(0, np.zeros(num_classes).tolist(), num_hps, num_classes, Exact_Table, 0)
    Exact_Table, _ = votes_to_class(0, np.zeros(num_classes).tolist(), hps_test, num_classes, Exact_Table, 0)
    print('Done')

    # for hp in range(num_hps):
    for hp in range(hps_test):
        x = SVM_separate_table["bias hp"+str(hp)]
        min_x = value_info["hp " + str(hp)]["min"]
        max_x = value_info["hp " + str(hp)]["max"]

        Exact_Table['threshold hp' + str(hp)] =   -math.floor(scale*((num_features + 1) * min_x))
        Exact_Table["bias hp" + str(hp)] =  math.floor(scale*(x - min_x))

    for i,fn in enumerate(feature_names):
        Exact_Table[fn] = {}
        for feature in range(feature_max[i]+1):
            Exact_Table[fn][feature] = {}
            # for hp in range(num_hps):
            for hp in range(hps_test):
                x = SVM_separate_table[fn][feature]["hp "+str(hp)]
                min_x = value_info["hp " + str(hp)]["min"]
                max_x = value_info["hp " + str(hp)]["max"]

                Exact_Table[fn][feature]["hp "+str(hp)] = math.floor(scale*(x - min_x))

    # =================== convert model timer ===================
    config['timer log']['convert model']['end'] = time.time()
    # =================== convert model timer ===================

    table_name = 'exact_table.json'
    json.dump(Exact_Table, open(f'eval/tables/{cur_dataset}/{cur_model}/{cur_trace}-{cur_model}-'
                                f'{model_size}-{table_name}', 'w'), indent=4)

    feature_tbl_len = []
    for f in range(num_features):
        feature_tbl_len += [len(Exact_Table['f'+str(f)].keys())]

    thresh_and_bias = ''
    # for h in range(num_hps):
    for h in range(hps_test):
        if h == 0:
            thresh_and_bias += str(Exact_Table['threshold hp' + str(h)])
        else:
            thresh_and_bias += (', '+ str(Exact_Table['threshold hp' + str(h)]))
    # for h in range(num_hps):
    for h in range(hps_test):
        thresh_and_bias += (', ' + str(Exact_Table['bias hp' + str(h)]))

    hp_info = {}
    count = 0
    initial = 0
    while True:
        for c in range(num_classes):
            if c > initial:
                hp_info[count] = [initial, c]
                count += 1
        initial += 1
        if initial >= num_classes - 1:
            break

    config['p4 config'] = {}

    config['p4 config'] = {}
    config['p4 config']["model"] = "SVM"
    config['p4 config']["number of features"] = num_features
    config['p4 config']["number of classes"] = num_classes
    config['p4 config']["action data bits"] = num_bits
    # config['p4 config']["number of hps"] = num_hps
    config['p4 config']["number of hps"] = hps_test
    config['p4 config']["feature tbl len"] = feature_tbl_len
    config['p4 config']['table name'] = f'{cur_trace}-{cur_model}-{model_size}-exact_table.json'
    config['p4 config']['thresh and bias'] = thresh_and_bias
    config['p4 config']['hp_info'] = hp_info

    config['test config'] = {}
    config['test config']['type of test'] = 'classification'

    json.dump(config,
              open(config['directory config']['work']+"/"+config_path, 'w'),
              indent=4,
              cls=NpEncoder)

    return sklearn_y_predict.tolist()

# def process_packets(batch, test_X, num_features, num_classes, Exact_Table, config,
#                     test_y, sklearn_test_y):
#     results = []
#     core_id = os.getpid()

#     hp_info = {}
#     count = 0
#     initial = 0
#     while True:
#         for c in range(num_classes):
#             if c> initial:
#                 hp_info[count] = [initial,c]
#                 count+=1
#         initial+=1
#         if initial >= num_classes-1:
#             break

#     for i in batch:
#         class_vote = np.zeros(num_classes).tolist()
#         input_feature_value = test_X.values[i]
#         print(num_features)
#         print(i)
#         print(input_feature_value)
#         for hp in range(int(num_classes * (num_classes - 1) / 2)):
#             hp_value = 0
#             for f in range(num_features):
#                 hp_value += Exact_Table["f"+str(f)][str(input_feature_value[f])]["hp "+str(hp)]
#             hp_value += Exact_Table["bias hp"+str(hp)]
#             if num_classes ==2:
#                 if hp_value>Exact_Table["threshold hp"+str(hp)]:
#                     class_vote[hp_info[hp][1]] += 1
#                 else:
#                     class_vote[hp_info[hp][0]] += 1
#             else:
#                 if hp_value>Exact_Table["threshold hp"+str(hp)]:
#                     class_vote[hp_info[hp][0]] += 1
#                 else:
#                     class_vote[hp_info[hp][1]] += 1

#         switch_prediction   = class_vote.index(np.max(class_vote))
#         # print(f'class vote: {class_vote}')
#         # print(f'switch pred: {switch_prediction}')
#         switch_proba        = sum(class_vote) / len(class_vote)

#         results.append((switch_prediction, switch_proba))

#     cur_ts = datetime.now()
#     cur_ts = cur_ts.strftime("%H-%M-%S")
#     print(f'[{cur_ts}]  Core {core_id} processed {len(batch)} packets.')
#     return results

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

    same            = 0
    correct         = 0
    error           = 0

    switch_test_y           = []
    switch_test_y_proba     = []

    hp_info = {}
    count   = 0
    initial = 0

    while True:
        for c in range(num_classes):
            if c > initial:
                hp_info[count] = [initial,c]
                count += 1
        initial += 1
        if initial >= num_classes-1:
            break

    for i in range(np.shape(test_X.values)[0]):
        class_vote = np.zeros(num_classes).tolist()
        input_feature_value = test_X.values[i]
        for hp in range(int(num_classes * (num_classes - 1) / 2)):
            hp_value = 0
            for f in range(num_features):
                hp_value += Exact_Table["f"+str(f)][str(input_feature_value[f])]["hp "+str(hp)]
            hp_value += Exact_Table["bias hp"+str(hp)]
            if num_classes == 2:
                if hp_value > Exact_Table["threshold hp"+str(hp)]:
                    class_vote[hp_info[hp][1]] += 1
                else:
                    class_vote[hp_info[hp][0]] += 1
            else:
                if hp_value>Exact_Table["threshold hp"+str(hp)]:
                    class_vote[hp_info[hp][0]] += 1
                else:
                    class_vote[hp_info[hp][1]] += 1

        switch_prediction = class_vote.index(np.max(class_vote))

        switch_test_y_proba += [sum(class_vote)/len(class_vote)]
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
                 cur_dataset,
                 cur_trace,
                 cur_model,
                 model_size,
                 'switch')

# def test_tables(sklearn_test_y, test_X, test_y, cur_dataset, cur_trace,
#                 config_path=None, threshold=None):
#     if config_path:
#         print(config_path)
#         config = json.load(open(config_path, 'r'))
#     else:
#         config = json.load(open('conf/planter_config.json', 'r'))

#     num_features    = config['data config']['number of features']
#     num_classes     = config['model config']['number of classes']
#     cur_model       = config['model config']['model']
#     model_size      = config['model config']['model size']
#     Exact_Table     = json.load(open(f'eval/tables/{cur_dataset}/{cur_model}/{cur_trace}-'
#                                      f'{cur_model}-{model_size}-exact_table.json', 'r'))

#     print("Test the generated table")
#     same = 0
#     correct = 0
#     error = 0
#     switch_test_y = []
#     switch_test_y_proba = []

#     batch_size = 10000

#     # Create batches of indices
#     indices = list(range(np.shape(test_X.values)[0]))
#     batches = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]

#     with ProcessingPool() as pool:
#         results = pool.map(lambda batch: process_packets(batch, test_X, num_features, num_classes, Exact_Table, config, test_y, sklearn_test_y), batches)

#         for batch_results in results:
#             for i, (switch_prediction, switch_proba) in enumerate(batch_results):
#                 switch_test_y.append(switch_prediction)
#                 switch_test_y_proba.append(switch_proba)

#                 # Calculate correct, same, and error counts
#                 original_index = batches[i // batch_size][i % batch_size]  # Get the original index
#                 if switch_prediction == test_y[original_index]:
#                     correct += 1

#                 if switch_prediction == sklearn_test_y[original_index]:
#                     same += 1
#                 else:
#                     error += 1

#     eval_metrics(test_y,
#                  switch_test_y,
#                  switch_test_y_proba,
#                  cur_dataset,
#                  cur_trace,
#                  cur_model,
#                  model_size,
#                  'switch')

def resource_prediction(config_path):
    config = json.load(open(config_path, 'r'))

    print('Exact match entries:     ', np.sum(config['p4 config']["code table size"]) \
          + config['p4 config']["decision table size"] )
