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

from src.functions.json_encoder import NpEncoder
from torch.autograd import Variable as V
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import torch.nn as nn
import torch
import json
import copy
import time


###### Define an autoencoder model
class autoencoder(nn.Module):
    def __init__(self, num_features, num_components):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
                nn.Linear(num_features, num_components),
                # nn.Tanh(),
                # nn.Linear(3, 2),
        )
        self.decoder = nn.Sequential(
                # nn.Linear(2, 3),
                # nn.Tanh(),
                nn.Linear(num_components, num_features),
                # nn.Sigmoid()
        )

    def forward(self, x):
        encoder = self.encoder(x)
        decoder = self.decoder(encoder)
        return encoder, decoder

def run_model(train_X, train_y, test_X, test_y, used_features, cur_dataset,
              cur_trace, config_path=None):
    cur_X = pd.concat([train_X, test_X])
    cur_y = np.concatenate((train_y, test_y), axis=None)

    last_n = cur_dataset[-3:]
    if last_n == '-ad':
        cur_dataset = cur_dataset [:-3]

    if config_path:
        print(f'Config: {config_path}')
        config = json.load(open(config_path, 'r'))
    else:
        config = json.load(open('conf/planter_config.json', 'r'))

    num_bits        = config['model config']['number of bits']
    num_components  = config['model config']['num components']
    num_features    = config['data config']['number of features']
    cur_model       = config['model config']['model']
    model_size      = config['model config']['model size']
    num_classes     = config['model config']['number of classes']
    learning_rate   = config['model config']['learning rate']
    batch_size      = config['model config']['batch size']
    num_epoch       = config['model config']['num epoch']

    feature_names = []
    for i, f in enumerate(used_features):
        train_X.rename(columns={f: "f" + str(i)}, inplace=True)
        test_X.rename(columns={f: "f" + str(i)}, inplace=True)
        feature_names += ["f" + str(i)]

    feature_max = []
    for i in feature_names:
        t_t = [test_X[[i]].max()[0], train_X[[i]].max()[0]]
        feature_max += [max(t_t)+1]
     ###### Normalize the input as the autoencoder only uses the input
    # MMScaler = MinMaxScaler()
    # x = MMScaler.fit_transform(x)
    # iforestX = x

    ###### Convert input data to the dataset type accepted by the neural network,
    # set batch size to 10

    tensor_x = torch.from_numpy(train_X.to_numpy().astype(np.float32))
    tensor_y = torch.from_numpy(train_y.astype(np.float32))

    # tensor_x = torch.from_numpy(train_X.to_numpy().astype(np.float32))
    # tensor_y = torch.from_numpy(train_y.astype(np.float32))
    # X_new = copy.deepcopy(test_X)
    # sklearn_X_new = copy.deepcopy(test_X)

    sklearn_X_new = copy.deepcopy(cur_X)

    test_X = torch.from_numpy(cur_X.to_numpy().astype(np.float32))
    test_y = torch.from_numpy(cur_y.astype(np.float32))

    dataset_train   = TensorDataset(tensor_x, tensor_y)
    dataset_test    = TensorDataset(test_X, test_y)

    dataset_train_loader    = DataLoader(dataset_train, batch_size=batch_size, shuffle=False)
    dataset_test_loader     = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    model = autoencoder(num_features, num_components)

    ####### Define the loss function

    criterion = nn.MSELoss()

    ####### Define the optimization function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # If using SGD, convergence does not decrease

    # =================== train model timer ===================
    config['timer log']['train model'] = {}
    config['timer log']['train model']['start'] = time.time()
    # =================== train model timer ===================

    ####### Set epoch to 300

    for epoch in range(num_epoch):
        total_loss = 0
        for i, (x, y) in enumerate(dataset_train_loader):
            _, pred = model(V(x))
            loss    = criterion(pred, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss
        # if epoch % 10 == 0:
        print(f'[Train] Cur epoch: {epoch}')
        print('[Train] Training loss {}'.format(total_loss.data.numpy()))

    # =================== train model timer ===================
    config['timer log']['train model']['end'] = time.time()
    # =================== train model timer ===================

    # =================== convert model timer ===================
    config['timer log']['convert model'] = {}
    config['timer log']['convert model']['start'] = time.time()
    # =================== convert model timer ===================

    model_info = {}
    for i, param in enumerate (model.parameters()):
        model_info[i] = param.detach().numpy()
    model_info['weights']   = model_info[0].T
    model_info['bias']      = model_info[1]
    ###### Perform dimensionality reduction and visualization based on the trained model

    print('\nGenerate the table...',end="")

    value_info = {}
    value_info["max"] = np.max(model_info['bias'])
    value_info["min"] = np.min(model_info['bias'])
    for ax in range(num_components):
        value_info["ax " + str(ax)] = {}
        value_info["ax " + str(ax)]["max"] = model_info['bias'][ax]
        value_info["ax " + str(ax)]["min"] = model_info['bias'][ax]

    g_table = {}
    for f in range(num_features):
        g_table['feature ' + str(f)] = {}
        for input_value in range(feature_max[f]):
            g_table['feature ' + str(f)][input_value] = {}
            for ax in range(num_components):
                middle_value = copy.deepcopy(input_value * model_info['weights'][f, ax])
                g_table['feature ' + str(f)][input_value]['ax' + str(ax)] = middle_value
                if middle_value > value_info["ax " + str(ax)]["max"]:
                    value_info["ax " + str(ax)]["max"] = middle_value
                if middle_value < value_info["ax " + str(ax)]["min"]:
                    value_info["ax " + str(ax)]["min"] = middle_value
                if middle_value > value_info["max"]:
                    value_info["max"] = middle_value
                if middle_value < value_info["min"]:
                    value_info["min"] = middle_value

    if num_bits != 0:
        scale = (2 ** num_bits) / ((value_info["max"] - value_info["min"]) * (num_features+1))

    Exact_Table = {}
    for f in range(num_features):
        Exact_Table['feature ' + str(f)] = {}
        for input_value in range(feature_max[f]):
            Exact_Table['feature ' + str(f)][input_value] = {}
            for ax in range(num_components):
                middle_value = copy.deepcopy(g_table['feature ' + str(f)][input_value]['ax' + str(ax)])
                if num_bits != 0:
                    middle_value = int(np.floor((middle_value - value_info["min"])*scale))
                Exact_Table['feature ' + str(f)][input_value]['ax' + str(ax)] = middle_value

    Exact_Table['bias'] = {}
    for ax in range(num_components):
        if num_bits != 0:
            Exact_Table['bias']['ax' + str(ax)] = int(np.floor((model_info['bias'][ax]- value_info["min"])*scale))
        else:
            Exact_Table['bias']['ax' + str(ax)] = int(np.floor((model_info['bias'][ax]- value_info["min"])))

    # =================== convert model timer ===================
    config['timer log']['convert model']['end'] = time.time()
    # =================== convert model timer ===================

    print('Done')
    json.dump(Exact_Table, open(f'eval/tables/{cur_dataset}/{cur_model}/{cur_trace}-{cur_model}-'
                                f'{model_size}-exact_table.json', 'w'), indent=4)

    feature_tbl_len = []
    for f in range(num_features):
        feature_tbl_len += [len(Exact_Table['feature ' + str(f)].keys())]

    config['p4 config']                         = {}
    config['p4 config']["model"]                = "ae"
    config['p4 config']["number of features"]   = num_features
    config['p4 config']["number of classes"]    = num_classes
    config['p4 config']["action data bits"]     = num_bits
    config['p4 config']['table name']           = f'{cur_trace}-{cur_model}-{model_size}-exact_table.json'
    config['p4 config']["feature tbl len"]      = feature_tbl_len
    config['p4 config']["num components"]       = num_components
    config['test config']                       = {}
    config['test config']['type of test']       = 'dimension_reduction'


    json.dump(config,
              open(config['directory config']['work']+"/"+config_path, 'w'),
              indent=4,
              cls=NpEncoder)

    x_ = []
    y_ = []
    for i, (x, y) in enumerate(dataset_test):
        _, pred = model(V(x))
        dimension = _.data.numpy()
        for ax in range(num_components):
            sklearn_X_new.values[i,ax] = dimension[ax] - (num_features + 1) * value_info["min"]
        # prepare for plot
        x_.append(dimension[0]-(num_features+1)*value_info["min"])
        y_.append(dimension[1]-(num_features+1)*value_info["min"])

    # plot_result =  input('- Plot the training result ? (default = n) ') or 'n'

    # if plot_result == 'y':
    #     print('plot')
    #     plt.scatter(numpy.array(x_), numpy.array(y_), c=test_y.detach().numpy())

    #     for i in range(len(numpy.array(x_))):
    #         plt.annotate(i, (x_[i], y_[i]))
    #     plt.show()

    return sklearn_X_new.values

def test_tables(sklearn_y, train_X, train_y, test_X, test_y, cur_dataset, cur_trace,
                config_path=None, threshold=None):
    cur_X = pd.concat([train_X, test_X])

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
            switch_y[i][ax] = copy.deepcopy(Exact_Table['bias']['ax'+str(ax)])
        for f in range(num_features):
            ax_middle = Exact_Table["feature "+str(f)][str(input_feature_value[f])]
            for ax in range(num_components):
                switch_y[i][ax] += ax_middle["ax"+str(ax)]
        # print(sklearn_test_x[i], switch_test_x[i])
        # test_X.values[i]
        # switch_test_x.values[i]

    switch_y = switch_y[:, :num_components]

    cur_ae_feats = pd.DataFrame(switch_y, columns=['ae_0', 'ae_1'])
    cur_ae_feats.to_csv(f'eval/feats/{cur_dataset}/{cur_model}/{cur_trace}-'
                        f'{cur_model}-{model_size}-feats.csv', index=False)

    # for ax in range(num_components):
    #     corr, _ = pearsonr(sklearn_test_x[:, ax],switch_test_x[:, ax])
    #     print('Pearsons correlation of M/A PCA and output of Pytorch for axis '+str(ax)+' is: %.4f' % corr)

def resource_prediction(config_path):
    config = json.load(open(config_path, 'r'))

    print('Exact match entries:     ', np.sum(config['p4 config']["code table size"]) \
          + config['p4 config']["decision table size"] )
