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

from src.functions.logic_gates import XNOR_with_bits
from src.functions.Range_to_TCAM_Top_Down import ten_to_bin
from src.functions.json_encoder import NpEncoder
from torch.utils.data import DataLoader, TensorDataset
from src.models.nn.BinaryNet.models.xnor_mlp import mlp
from src.models.nn.BinaryNet.classifiers.xnor_classifier import XnorClassifier
from eval.eval_metrics import eval_metrics
import math
import numpy as np
import time
import copy
import json
import torch


def bintoint(binary):
    number = 0

    for b in binary:
        number = (2 * number) + int(b)

    return number

def convert_weight_to_register_data(weight_data):
    weight  = []
    weights = []

    for i in weight_data:
        for j in i:
            if j < 0:
                weight.append(0)
            else:
                weight.append(1)

        weights.append(bintoint(weight))
        weight.clear()

    return weights

def run_model(train_X, train_y, test_X, test_y, used_features, cur_dataset,
              cur_trace, config_path=None):
    if config_path:
        print(f'Config: {config_path}')
        config = json.load(open(config_path, 'r'))
    else:
        config = json.load(open('conf/planter_config.json', 'r'))

    num_features        = config['data config']['number of features']
    num_layers          = config['model config']['number of layers']
    num_hidden_nodes    = config['model config']['num hidden nodes']
    num_classes         = config['model config']['number of classes']
    learning_rate       = config['model config']['learning rate']
    batch_size          = config['model config']['batch size']
    num_epoch           = config['model config']['num epoch']

    feature_names = []
    for i, f in enumerate(used_features):
        train_X.rename(columns={f: "f" + str(i)}, inplace=True)
        test_X.rename(columns={f: "f" + str(i)}, inplace=True)
        feature_names += ["f"+str(i)]

    feature_max = []
    width       = []

    for i in feature_names:
        t_t             = [test_X[[i]].max()[0], train_X[[i]].max()[0]]
        feature_max    += [np.max(t_t)+1]

    for f in range(num_features):
        width += [np.ceil(math.log(feature_max[f],2))]

    width_row   = int(np.sum(width))
    total_count = np.shape(train_X.values)[0] + np.shape(test_X.values)[0]

    count       = 0
    train_X_new = []
    test_X_new  = []

    for i in range(np.shape(train_X.values)[0]):
        flag    = 0
        row     = (np.zeros(int(width_row)))

        for f in range(num_features):
            code = ten_to_bin(train_X.values[i][f],width[f])

            for d in range(int(width[f])):
                row[flag]   = int(code[d])
                flag       += 1

        train_X_new += [row]
        count       += 1
        percent     = int(np.ceil(50 * count / total_count))

        print('\rProcessing the raw Data [' + percent * '#' + (50 - percent) * '-' + '] ' \
              + str( int(np.round(100 * count / total_count))) + "%", end="")

    train_X_new = np.array(train_X_new)

    for i in range(np.shape(test_X.values)[0]):
        flag    = 0
        row     = (np.zeros(int(width_row)))

        for f in range(num_features):
            code = ten_to_bin(test_X.values[i][f],width[f])

            for d in range(int(width[f])):
                row[flag] = int(code[d])
                flag     += 1

        test_X_new += [row]
        count      += 1
        percent     = int(np.ceil(50 * count / total_count))

        print('\rProcessing the raw data [' + percent * '#' + (50 - percent) * '-' + '] ' \
              + str(int(np.round(100 * count / total_count))) + "%", end="")

    test_X_new = np.array(test_X_new)

    print('\nData set is ready')

    # Convert input data to the dataset type accepted by the neural network, set batch size to 10
    tensor_x            = torch.from_numpy(train_X_new.astype(np.float32))
    tensor_y            = torch.LongTensor(train_y.astype(np.float32))
    test_X              = torch.from_numpy(test_X_new.astype(np.float32))
    test_y_tensor       = torch.LongTensor(test_y.astype(np.float32))
    my_train_dataset    = TensorDataset(tensor_x, tensor_y)
    my_test_dataset     = TensorDataset(test_X, test_y_tensor)
    train_loader        = DataLoader(my_train_dataset, batch_size=batch_size, shuffle=False)
    test_loader         = DataLoader(my_test_dataset, batch_size=batch_size, shuffle=False)

    cuda    = torch.cuda.is_available()
    device  = torch.device('cuda' if cuda else 'cpu')

    torch.manual_seed(0)

    if cuda:
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(0)

    # =================== train model timer ===================
    config['timer log']['train model']          = {}
    config['timer log']['train model']['start'] = time.time()
    # =================== train model timer ===================

    model = eval('mlp')(width_row, num_hidden_nodes, num_layers, num_classes)
    model.to(device)

    classification = XnorClassifier(model, train_loader, test_loader, device)

    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(device)

    if hasattr(model, 'init_w'):
        model.init_w()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [100, 200] , gamma=0.1)

    classification.train(criterion, optimizer, num_epoch, scheduler,
                         config['directory config']['work']+'/src/tmp/mlp')

    nn_output = classification.test(criterion, True)

    # nn_pred = nn_output[1]
    nn_prob = nn_output[2]
    nn_pred_bin = np.where(nn_prob < 0.5, 0, 1)

    eval_metrics(test_y,
                 nn_pred_bin,
                 nn_prob,
                 f'nn-{num_layers}-{num_hidden_nodes[0]}-{learning_rate}-{batch_size}-{num_epoch}-pytorch',
                 cur_dataset,
                 cur_trace)

    # =================== train model timer ===================
    config['timer log']['train model']['end'] = time.time()
    # =================== train model timer ===================

    # =================== convert model timer ===================
    config['timer log']['convert model']            = {}
    config['timer log']['convert model']['start']   = time.time()
    # =================== convert model timer ===================

    Exact_Table             = {}
    Exact_Table['weights']  = []

    for a in range(num_layers):
        Exact_Table['weights'] += convert_weight_to_register_data(model.classifier._modules['layer'+str(a)].weight.detach().numpy())

    # =================== convert model timer ===================
    config['timer log']['convert model']['end'] = time.time()
    # =================== convert model timer ===================

    json.dump(Exact_Table, open(f'tables/{cur_dataset}-{cur_trace}-nn-'
                                f'{num_layers}-{num_hidden_nodes[0]}-{learning_rate}-'
                                f'{batch_size}-{num_epoch}-exact_table.json', 'w'), indent=4)

    print('table_exact is generated')

    config['p4 config']                         = {}
    config['p4 config']["model"]                = "nn"
    config['p4 config']["num hidden nodes"]     = num_hidden_nodes
    config['p4 config']["number of features"]   = num_features
    config['p4 config']["number of layers"]     = num_layers
    config['p4 config']["number of classes"]    = num_classes
    config['p4 config']["width of inputs"]      = width
    config['p4 config']['table name']           = f'{cur_dataset}-{cur_trace}-nn-{num_layers}-{num_hidden_nodes[0]}-{learning_rate}-{batch_size}-{num_epoch}-exact_table.json'
    config['test config']                       = {}
    config['test config']['type of test']       = 'classification'

    json.dump(config,
              open(config['directory config']['work'] + '/' + config_path, 'w'),
              indent=4,
              cls=NpEncoder)

    return test_y.tolist()

def bits_on_count(x):
  return sum(c=='1' for c in bin(x))

def test_tables(sklearn_test_y, test_X, test_y, cur_dataset, cur_trace,
                config_path=None, threshold=None):
    if config_path:
        print(config_path)
        config = json.load(open(config_path, 'r'))
    else:
        config = json.load(open('conf/planter_config.json', 'r'))

    num_features        = config['data config']['number of features']
    num_classes         = config['model config']['number of classes']
    num_hidden_nodes    = config['p4 config']["num hidden nodes"]
    num_layers          = config['p4 config']["number of layers"]
    width               = config['p4 config']["width of inputs"]
    learning_rate       = config['model config']['learning rate']
    batch_size          = config['model config']['batch size']
    num_epoch           = config['model config']['num epoch']

    Exact_Table = json.load(open(f'tables/{cur_dataset}-{cur_trace}-nn-'
                                 f'{num_layers}-{num_hidden_nodes[0]}-{learning_rate}-'
                                 f'{batch_size}-{num_epoch}-exact_table.json', 'r'))

    print('Test the exact feature table, extact code and decision table (feel free if the acc to sklearn is slightly lower than 1)')

    correct         = 0
    switch_test_y   = []
    switch_prob     = []

    for i in range(np.shape(test_X.values)[0]):
        if i % 10000 == 0:
            print(f'Test: processed {i} packets.')
        input = ''
        for f in range(num_features):
            input += ten_to_bin(test_X.values[i][f],width[f])
        input = int(input, 2)
        node_num = 0
        for a in range(num_layers):
            if a == 0:
                num_bits = int(np.sum(width))
            else:
                num_bits = int(num_hidden_nodes[a - 1])

            next_layer_input = ''

            if a + 1 != num_layers:
                for n in range(num_hidden_nodes[a]):
                    value = XNOR_with_bits(input, Exact_Table['weights'][node_num], num_bits)
                    value = bits_on_count(value)

                    node_num += 1

                    if a == 0:
                        threshold = np.floor(np.sum(width)/2)
                    else:
                        threshold = np.floor(num_hidden_nodes[a-1]/2)

                    if value > threshold:
                        next_layer_input += '1'
                    else:
                        next_layer_input += '0'

                input = int(next_layer_input,2)

            else:
                result = np.zeros(num_classes).tolist()

                for c in range(num_classes):
                    value       = XNOR_with_bits(input, Exact_Table['weights'][node_num], num_bits)
                    value       = bits_on_count(value)
                    result[c]   = copy.deepcopy(value)
                    node_num   += 1

        logits              = torch.tensor(result).float()
        probabilities       = torch.softmax(logits, dim=0)
        max_value, max_idx  = torch.max(probabilities, dim=0)
        switch_prediction   = result.index(np.max(result))
        switch_prob        += [max_value.item()]
        switch_test_y      += [switch_prediction]

        if switch_prediction == test_y[i]:
            correct += 1

    eval_metrics(test_y,
                 switch_test_y,
                 switch_prob,
                 f'nn-{num_layers}-{num_hidden_nodes[0]}-{learning_rate}-{batch_size}-{num_epoch}-switch',
                 cur_dataset,
                 cur_trace)
