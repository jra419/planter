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

from plugins.planter.planter_fork.src.functions.json_encoder import NpEncoder
from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import pandas as pd
import numpy as np
import statistics
import json
import copy
import os


class PCA_:
    def __init__(self, conf):
        self.conf       = json.load(open(conf, 'r'))
        self.conf_path  = conf

        self.num_components = self.conf['model config']['num components']
        self.pca_train      = []
        self.value_info     = {}
        self.last_pca_pred  = []

        self.pca    = PCA(n_components=self.num_components)
        # self.scaler = StandardScaler()

        # self.corr_aggr_0 = []
        # self.corr_aggr_1 = []

        self.corr       = []
        self.corr_aggr  = []

    def train_model(self, train_x, switch_model, num_features=80):
        cur_dataset     = self.conf['data config']['dataset']
        cur_trace       = self.conf['data config']['cur_trace']
        num_features    = self.conf['data config']['number of features']
        cur_model       = self.conf['model config']['model']
        model_size      = self.conf['model config']['model size']
        num_bits        = self.conf['model config']['number of bits']
        num_components  = self.conf['model config']['num components']
        num_classes     = self.conf['model config']['number of classes']

        # self.scaler.fit(train_x)
        # train_x = self.scaler.transform(train_x)

        train_x = pd.DataFrame(train_x)

        new_column_names    = [f"f{i}" for i in range(num_features)]
        train_x.columns       = new_column_names

        sklearn_x_new   = self.pca.fit_transform(train_x.values)

        feat_max = []
        for i in new_column_names:
            t_t_max = [train_x[[i]].max()[0]]
            feat_max += [np.max(t_t_max)+1]

        model_info                  = {}
        model_info['means']         = self.pca.mean_
        model_info['components']    = self.pca.components_.T

        self.value_info["max"]   = 0
        self.value_info["min"]   = 0

        for ax in range(num_components):
            self.value_info["ax "+str(ax)]           = {}
            self.value_info["ax " + str(ax)]["max"]  = 0
            self.value_info["ax " + str(ax)]["min"]  = 0

        if switch_model:
            PCA_Table = {}

            for f in range(num_features):
                PCA_Table['feature '+str(f)] = {}

                for input_value in range(int(feat_max[f])):
                    PCA_Table['feature ' + str(f)][input_value] = {}
                    value = input_value - model_info['means'][f]

                    for ax in range(num_components):
                        middle_value = copy.deepcopy(value*model_info['components'][f,ax])
                        PCA_Table['feature ' + str(f)][input_value]['ax'+str(ax)] = middle_value

                        if middle_value > self.value_info["ax " + str(ax)]["max"]:
                            self.value_info["ax " + str(ax)]["max"] = middle_value
                        if middle_value < self.value_info["ax " + str(ax)]["min"]:
                            self.value_info["ax " + str(ax)]["min"] = middle_value
                        if middle_value > self.value_info["max"]:
                            self.value_info["max"] = middle_value
                        if middle_value < self.value_info["min"]:
                            self.value_info["min"] = middle_value

            if num_bits != 0:
                scale = (2**num_bits)/((self.value_info["max"]-self.value_info["min"])*(num_features))

            Exact_Table = {}

            for f in range(num_features):
                Exact_Table['feature ' + str(f)] = {}

                for input_value in range(int(feat_max[f])):
                    Exact_Table['feature ' + str(f)][input_value] = {}

                    for ax in range(num_components):
                        middle_value = copy.deepcopy(PCA_Table['feature ' + str(f)][input_value]['ax' + str(ax)])
                        if num_bits != 0:
                            middle_value = int(np.floor((middle_value - self.value_info["min"])*scale))
                        Exact_Table['feature ' + str(f)][input_value]['ax' + str(ax)] = middle_value

            outdir = f'eval/planter/{cur_dataset}/{cur_model}/tables'
            if not os.path.exists(f'eval/planter/{cur_dataset}/{cur_model}/tables'):
                os.makedirs(outdir, exist_ok=True)
            outpath_exact_table = os.path.join(outdir, f'{cur_trace}-{cur_model}-{model_size}-exact_table.json')
            json.dump(Exact_Table, open(outpath_exact_table, 'w'), indent=4)

            feature_tbl_len = []

            for f in range(num_features):
                feature_tbl_len += [len(Exact_Table['feature ' + str(f)].keys())]

            self.conf['p4 config']                         = {}
            self.conf['p4 config']["model"]                = "PCA"
            self.conf['p4 config']["number of features"]   = num_features
            self.conf['p4 config']["number of classes"]    = num_classes
            self.conf['p4 config']["action data bits"]     = num_bits
            self.conf['p4 config']['table name']           = (
                f'{cur_trace}-{cur_model}-{model_size}-exact_table.json')
            self.conf['p4 config']["feature tbl len"]      = feature_tbl_len
            self.conf['p4 config']["num components"]       = num_components
            self.conf['test config']                       = {}
            self.conf['test config']['type of test']       = 'dimension_reduction'

            json.dump(self.conf, open(self.conf_path, 'w'), indent=4, cls=NpEncoder)

        self.pca_train = copy.deepcopy(sklearn_x_new)

        for ax in range(num_components):
            self.pca_train[:, ax] = sklearn_x_new[:, ax] - num_features*(self.value_info["min"])

    def train_pred(self, test_x, num_features=80):
        test_x              = pd.DataFrame(test_x)
        new_column_names    = [f"f{i}" for i in range(num_features)]
        test_x.columns      = new_column_names

        # test_x = self.scaler.transform(test_x.values)

        sklearn_x_new = self.pca.transform(test_x.values)

        pca_pred = copy.deepcopy(sklearn_x_new)

        for ax in range(self.num_components):
            pca_pred[:, ax] = sklearn_x_new[:, ax] - num_features*(self.value_info["min"])

        self.last_pca_pred = pca_pred

        return pca_pred

    def test_tables(self, test_x, num_features=80):
        num_features    = self.conf['data config']['number of features']
        cur_dataset     = self.conf['data config']['dataset']
        cur_trace       = self.conf['data config']['cur_trace']
        cur_model       = self.conf['model config']['model']
        model_size      = self.conf['model config']['model size']
        num_components  = self.conf['model config']['num components']

        Exact_Table     = json.load(open(f'eval/planter/{cur_dataset}/{cur_model}/tables/{cur_trace}-{cur_model}-{model_size}-exact_table.json', 'r'))

        test_x              = pd.DataFrame(test_x)
        new_column_names    = [f"f{i}" for i in range(num_features)]
        test_x.columns      = new_column_names

        # test_x = self.scaler.transform(test_x.values)

        pca_pred_switch = copy.deepcopy(self.last_pca_pred)

        for i in range(test_x.shape[0]):
            input_feature_value = test_x[i]
            for ax in range(num_components):
                pca_pred_switch[i][ax] = 0
            for f in range(num_features):
                try:
                    ax_middle = Exact_Table["feature "+str(f)][str(int(input_feature_value[f]))]
                except KeyError:
                    min_d = min(list(map(int, Exact_Table["feature "+str(f)].keys())))
                    max_d = max(list(map(int, Exact_Table["feature "+str(f)].keys())))
                    if int(input_feature_value[f]) < min_d:
                        ax_middle = Exact_Table["feature "+str(f)][str(min_d)]
                    else:
                        ax_middle = Exact_Table["feature "+str(f)][str(max_d)]

                for ax in range(num_components):
                    pca_pred_switch[i][ax] += ax_middle["ax"+str(ax)]

        for ax in range(num_components):
            corr, _ = pearsonr(self.last_pca_pred[:, ax], pca_pred_switch[:, ax])
            if ax + 1 > len(self.corr):
                self.corr.append([corr])
                self.corr_aggr.append(statistics.fmean(self.corr[ax]))
            else:
                self.corr[ax]      += [corr]
                self.corr_aggr[ax]  = statistics.fmean(self.corr[ax])

        return pca_pred_switch
