from sklearn.ensemble import IsolationForest
from plugins.planter.planter_fork.src.functions.json_encoder import NpEncoder
from plugins.planter.planter_fork.src.functions.Range_to_LPM import Table_to_LPM
from sklearn.tree import _tree
from collections import Counter
import pandas as pd
import numpy as np
import math
import json
import copy
import re
import gc

REUSE_TABLES = False
SKLEARN_ONLY = True

class IF:
    def __init__(self, conf):
        self.conf       = json.load(open(conf, 'r'))
        self.conf_path  = conf

        num_trees       = self.conf['model config']['number of trees']
        num_samples     = self.conf['model config']['number of samples']

        rng = np.random.RandomState(42)

        # fit the model
        self.clf = IsolationForest(n_estimators=num_trees,
                                   max_samples=num_samples,
                                   random_state=rng)

    def get_lineage(self, tree, feature_names, file):
        left            = tree.tree_.children_left
        right           = tree.tree_.children_right
        threshold       = tree.tree_.threshold
        features        = [feature_names[i] for i in tree.tree_.feature]
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

            ind     = n_node_samples[node]
            clause  = clause[:-4] + ' then ' + str(ind)

            file.write(clause)
            file.write(";\n")

    def print_tree(self, tree, feature_names):
        tree_           = tree.tree_
        feature_name    = [
                feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
                for i in tree_.feature
        ]
        share           = {}

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

    def ten_to_bin(self, num, count):
        num = bin(int(num)).lstrip('0b')
        if len(num) != count:
            cont    = count - len(num)
            num     = cont * '0' + num
        return num

    def find_feature_split(self, model, tree_index, num_features):
        feature_names = []
        feature_split = {}

        for a in range(num_features):
            feature_split["feature "+str(a)] = []
            feature_names                   += [f'f{a}']

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

        tree = open('plugins/planter/planter_fork/src/tmp/tree'+str(tree_index)+'.txt', "w+")

        for a in range(num_features):
            tree.write(str(feature_names[a]) + " = ")
            tree.write(str(feature_split["feature "+str(a)]))
            tree.write(";\n")

        self.print_tree(model, feature_names)
        self.get_lineage(model, feature_names, tree)
        tree.close()

        textfile = 'plugins/planter/planter_fork/src/tmp/tree'+str(tree_index)+'.txt'

        for f in range(num_features):
            feature_split['feature ' + str(f)] = sorted(list(set(feature_split['feature ' + str(f)])))

        return textfile, feature_split

    def generate_feature_tables(self, split, num_features,feature_max, table):
        for i in range(num_features):
            table["feature "+str(i)]    = {}
            count_code                  = 0
            nife                        = sorted(split["feature "+str(i)])

            table["feature " + str(i)][count_code] = 0
            if nife != []:
                for j, count_code in enumerate(nife):
                    if count_code < feature_max[i]-1:
                        table["feature " + str(i)][count_code+1] = j+1

        return table

    def find_classification(self, textfile, feature_split, num_features):
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

        classification  = []
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

            classification.append([num_nodes, int(num[i][a - 1])])

        return feature_n, classification

    def find_path_for_leaf_nodes(self, feature_n, classfication, num_features):
        path_to_leaf = {}

        for i in range(len(classfication)):
            path_to_leaf["path "+str(i)]            = {}
            path_to_leaf["path " + str(i)]["leaf"]  = classfication[i]

            for j in range(num_features):
                path_to_leaf["path " + str(i)]["feature "+str(j)] = feature_n[j][i]
        return path_to_leaf

    def generate_code_table_for_path(self, table, leaf_path, code_dict, feature_num,
                                     num_features, count):
        if feature_num == num_features:
            table['ctv'][count] = {}

            for f in range(num_features):
                table['ctv'][count]['f'+str(f)] = code_dict['feature ' + str(f)]

            table['ctv'][count]['leaf'] = leaf_path['leaf']
            count                      += 1

            gc.collect()
            return table, count
        else:
            for value in leaf_path['feature '+str(feature_num)]:
                code_dict['feature ' + str(feature_num)] = value
                feature_num += 1
                table, count = self.generate_code_table_for_path(
                        table, leaf_path, code_dict, feature_num, num_features, count)
                feature_num -= 1

        gc.collect()
        return table, count

    def generate_code_table(self, table, path_to_leaf, num_features):
        table['ctv']    = {}
        count           = 0

        for p in path_to_leaf:
            table, count = self.generate_code_table_for_path(
                    table, path_to_leaf[p], {}, 0, num_features, count)

        return table

    def generate_table(self, model, tree_index, num_features, g_table, feature_max, leaf_info):
        textfile, feature_split = self.find_feature_split(model, tree_index, num_features)

        g_table[tree_index] = {}
        g_table[tree_index] = self.generate_feature_tables(
                feature_split, num_features, feature_max, g_table[tree_index])

        feature_n, classification   = self.find_classification(
                textfile, feature_split, num_features)
        path_to_leaf                = self.find_path_for_leaf_nodes(
                feature_n, classification, num_features)

        code_width_for_feature = np.zeros(num_features)

        for i in range(num_features):
            code_width_for_feature[i] = int(np.ceil(math.log(g_table[tree_index]['feature ' + str(i)][np.max(list(g_table[tree_index]['feature ' + str(i)].keys()))]+1,2))) or 1

        g_table[tree_index] = self.generate_code_table(
                g_table[tree_index], path_to_leaf, num_features)

        print(classification)
        print('\rThe table for Tree: {} is generated'.format(tree_index))

        leaf_info['tree '+str(tree_index)]= np.unique(classification, axis=0)

        return g_table, leaf_info

    def _average_path_length(self, n_samples_leaf):
        """
        The average path length in a n_samples iTree, which is equal to
        the average path length of an unsuccessful BST search since the
        latter has the same structure as an isolation tree.
        Parameters
        ----------
        n_samples_leaf : array-like of shape (n_samples,)
            The number of training samples in each test sample leaf, for
            each estimators.

        Returns
        -------
        average_path_length : ndarray of shape (n_samples,)
        """

        # n_samples_leaf = check_array(n_samples_leaf, ensure_2d=False)

        n_samples_leaf_shape    = n_samples_leaf.shape
        n_samples_leaf          = n_samples_leaf.reshape((1, -1))
        average_path_length     = np.zeros(n_samples_leaf.shape)

        mask_1      = n_samples_leaf <= 1
        mask_2      = n_samples_leaf == 2
        not_mask    = ~np.logical_or(mask_1, mask_2)

        average_path_length[mask_1]     = 0.
        average_path_length[mask_2]     = 1.
        average_path_length[not_mask]   = (
                2.0 * (np.log(n_samples_leaf[not_mask] - 1.0) + np.euler_gamma)
                - 2.0 * (n_samples_leaf[not_mask] - 1.0) / n_samples_leaf[not_mask]
        )

        return average_path_length.reshape(n_samples_leaf_shape)

    def complex_list_idx(self, target_list, component):
        for i, x in enumerate(target_list):
            if np.all(x==component):
                return i

    def votes_to_class(self, tree_num, vote_list, num_trees, num_classes, g_table, num,
                    leaf_info, path_len_threshold):
        if tree_num  == num_trees:
            vote = 0

            for t in range(num_trees):
                vote += (leaf_info["tree "+str(t)][vote_list[t]][0]
                        + self._average_path_length(leaf_info["tree "+str(t)][vote_list[t]][1]))

            g_table['votes to class'][num] = {}

            for t in range(len(vote_list)):
                g_table['votes to class'][num]['t'+str(t)+' vote'] = leaf_info["tree "+str(t)][vote_list[t]]

            if vote >= path_len_threshold*num_trees:
                g_table['votes to class'][num]['class'] = 0
            else:
                g_table['votes to class'][num]['class'] = 1

            num += 1

            return g_table, num
        else:
            for value in range(len(leaf_info["tree "+str(tree_num)])):
                vote_list[tree_num] = value
                tree_num           += 1
                g_table, num        = self.votes_to_class(
                        tree_num, vote_list, num_trees, num_classes, g_table, num,
                        leaf_info, path_len_threshold)
                tree_num           -= 1
        return g_table, num

    def train_model(self, train_x, num_features=80):
        cur_dataset     = self.conf['data config']['dataset']
        cur_trace       = self.conf['data config']['cur_trace']
        cur_model       = self.conf['model config']['model']
        model_size      = self.conf['model config']['model size']
        num_classes     = self.conf['model config']['number of classes']
        num_trees       = self.conf['model config']['number of trees']
        num_samples     = self.conf['model config']['number of samples']

        train_x = pd.DataFrame(train_x)

        new_column_names = [f"f{i}" for i in range(num_features)]
        train_x.columns = new_column_names

        feat_max = []
        for i in new_column_names:
            t_t = [train_x[[i]].max()[0]]
            feat_max += [np.max(t_t)+1]

        self.clf.fit(train_x)

        path_len_thres = (2 * (np.log(num_samples - 1) + np.euler_gamma) - (2 * (num_samples - 1) / num_samples)) * (-math.log(0.5, 2))

        g_table                 = {}
        leaf_info               = {}
        leaf_info['max value']  = 0
        leaf_info['min value']  = 0

        for idx, estimator in enumerate(self.clf.estimators_):
            g_table, leaf_info = self.generate_table(
                    estimator, idx, num_features, g_table, feat_max, leaf_info)

        g_table['votes to class'] = {}
        print("\nGenerating vote to class table...")
        g_table, _ = self.votes_to_class(
                0, np.zeros(num_trees).tolist(), num_trees, num_classes, g_table, 0,
                leaf_info, path_len_thres)

        print('Done')

        json.dump(g_table, open(f'eval/tables/{cur_dataset}/{cur_model}/{cur_trace}-{cur_model}-'
                                f'{model_size}-g_table.json', 'w'), indent=4, cls=NpEncoder)

        for t in range(num_trees):
            leaf_info['tree ' + str(t)] = list(leaf_info['tree ' + str(t)])
            for i, x in enumerate(leaf_info['tree ' + str(t)]):
                tmp_list = list(x)
                tmp_list = [a.item() for a in tmp_list]
                leaf_info['tree ' + str(t)][i] = str(list(tmp_list))

        for t in range(num_trees):
            for k in g_table[t]['code to vote'].keys():
                g_table[t]['code to vote'][k]['leaf'] = leaf_info['tree ' + str(t)].index(str(list(g_table[t]['code to vote'][k]['leaf'])))

        for k in g_table['votes to class'].keys():
            for t in range(num_trees):
                tmp_list = list(g_table['votes to class'][k]['t'+str(t)+' vote'])
                tmp_list = [a.item() for a in tmp_list]
                g_table['votes to class'][k]['t'+str(t)+' vote'] = leaf_info['tree ' + str(t)].index(str(tmp_list))

        feat_width = []
        for max_f in feat_max:
            feat_width += [int(np.ceil(math.log(max_f, 2)) + 1)]

        code_width_tree_feature = np.zeros((num_trees, num_features))
        for i in range(num_features):
            for tree in range(num_trees):
                code_width_tree_feature[tree, i] = int(np.ceil(math.log(
                    g_table[tree]['feature ' + str(i)][np.max(list(g_table[tree]['feature ' + str(i)].keys()))] + 1, 2) + 1)) or 1

        LPM_Table = {}
        LPM_Table['decision'] = g_table['votes to class']

        for tree in range(num_trees):
            LPM_Table['tree ' + str(tree)] = g_table[tree]['code to vote']

        for i in range(num_features):
            LPM_Table['feature ' + str(i)] = {}
            for value in range(feat_max[i]):
                LPM_Table['feature ' + str(i)][value] = []
                for tree in range(num_trees):
                    LPM_Table['feature ' + str(i)][value] += [g_table[tree]["feature " + str(i)][value]]
        Exact_Table = copy.deepcopy(LPM_Table)
        for i in range(num_features):
            if i != 0:
                print('')
            print('Begin transfer: Feature table ' + str(i))
            LPM_Table['feature ' + str(i)] = Table_to_LPM(LPM_Table['feature ' + str(i)], feat_width[i])

        # ===================== prepare default vote =========================

        print("\nPreparing default vote...")
        collect_votes = []

        for t in range(num_trees):
            collect_votes.extend(int(Exact_Table['tree ' + str(t)][idx]['leaf']) for idx in Exact_Table['tree ' + str(t)])

        default_vote = Counter(collect_votes).most_common(1)[0][0]

        code_table_size = 0
        for t in range(num_trees):
            LPM_Table['tree ' + str(t)] = {}
            for idx in Exact_Table['tree ' + str(t)]:
                if int(Exact_Table['tree ' + str(t)][idx]['leaf']) != default_vote:
                    LPM_Table['tree ' + str(t)][code_table_size] = Exact_Table['tree ' + str(t)][idx]
                    code_table_size += 1
            Exact_Table['tree ' + str(t)] = copy.deepcopy(LPM_Table['tree ' + str(t)])
        print('Done')

        # ===================== prepare default class =========================

        print("Preparing default class...")

        collect_class = []
        collect_class = [value['class'] for value in Exact_Table['decision'].values()]
        default_class = Counter(collect_class).most_common(1)[0][0]

        code_table_size         = 0
        LPM_Table['decision']   = {}

        for idx in Exact_Table['decision']:
            if Exact_Table['decision'][idx]['class'] != default_class:
                LPM_Table['decision'][code_table_size] = Exact_Table['decision'][idx]
                code_table_size += 1

        Exact_Table['decision'] = copy.deepcopy(LPM_Table['decision'])
        print('Done')

        json.dump(LPM_Table, open(f'eval/tables/{cur_dataset}/{cur_model}/{cur_trace}-{cur_model}-'
                                f'{model_size}-lpm_table.json', 'w'), indent=4, cls=NpEncoder)

        json.dump(Exact_Table, open(f'eval/tables/{cur_dataset}/{cur_model}/{cur_trace}-{cur_model}-'
                                    f'{model_size}-exact_table.json', 'w'), indent=4, cls=NpEncoder)

        self.conf['p4 config']                         = {}
        self.conf['p4 config']["model"]                = "if"
        self.conf['p4 config']["number of features"]   = num_features
        self.conf['p4 config']["number of classes"]    = num_classes
        self.conf['p4 config']["number of trees"]      = num_trees
        self.conf['p4 config']['table name']           = f'{cur_trace}-{cur_model}-{model_size}-lpm_table.json'
        self.conf['p4 config']["decision table size"]  = len(LPM_Table['decision'].keys())
        self.conf['p4 config']["code table size"]      = []
        for tree in range(num_trees):
            self.conf['p4 config']["code table size"] += [len(LPM_Table['tree ' + str(tree)].keys())]
        self.conf['p4 config']["default vote"]         = default_vote
        self.conf['p4 config']["default label"]        = default_class
        self.conf['p4 config']["width of feature"]     = feat_width
        self.conf['p4 config']["width of code"]        = code_width_tree_feature
        self.conf['p4 config']["used columns"]         = []
        for i in range(num_features):
            self.conf['p4 config']["used columns"]    += [len(LPM_Table['feature ' + str(i)].keys())]
        self.conf['p4 config']["width of probability"] = 7
        self.conf['p4 config']["width of result"]      = 8
        self.conf['p4 config']["standard headers"]     = ["ethernet", "Planter", "arp", "ipv4",
                                                          "tcp", "udp", "vlan_tag"]
        self.conf['test config']                       = {}
        self.conf['test config']['type of test']       = 'classification'

        json.dump(self.conf, open(self.conf_path, 'w'), indent=4, cls=NpEncoder)

    def train_pred(self, test_x):
        for i, f in enumerate(len(test_x.columns)):
            test_x.rename(columns={f: "f" + str(i)}, inplace=True)

        y_pred_test         = self.clf.predict(test_x)

        sklearn_y_pred      = copy.deepcopy(y_pred_test)
        sklearn_y_scores    = (-1.0) * self.clf.decision_function(test_x)

        for i in range(len(y_pred_test)):
            if y_pred_test[i] == -1:
                sklearn_y_pred[i] = 1
            if y_pred_test[i] == 1:
                sklearn_y_pred[i] = 0

        return [sklearn_y_pred, sklearn_y_scores]

    def test_tables(self, test_x):

        cur_dataset     = self.conf['data config']['dataset']
        cur_trace       = self.conf['data config']['cur_trace']
        num_features    = self.conf['data config']['number of features']
        cur_model       = self.conf['model config']['model']
        model_size      = self.conf['model config']['model size']
        num_trees       = self.conf['model config']['number of trees']

        table_lpm       = json.load(open(f'eval/tables/{cur_dataset}/{cur_model}/{cur_trace}-'
                                        f'{cur_model}-{model_size}-lpm_table.json', 'r'))
        table_exact     = json.load(open(f'eval/tables/{cur_dataset}/{cur_model}/{cur_trace}-'

                                        f'{cur_model}-{model_size}-exact_table.json', 'r'))

        default_vote = self.conf['p4 config']["default vote"]
        default_pred = self.conf['p4 config']["default label"]

        switch_test_y       = []
        switch_test_y_proba = []

        sorted_keys = {f: set(np.sort(list(table_lpm['feature ' + str(f)].keys()))) for f in range(num_features)}

        for i in range(np.shape(test_x.values)[0]):
            vote_list       = np.zeros(num_trees).astype(dtype=int)
            input_feat_val  = test_x.values[i]
            anomaly_cnt     = 0

            for tree in range(num_trees):
                code_list           = np.zeros(num_features, dtype=int)
                lpm_code_list       = np.zeros(num_features, dtype=int)

                for f in range(num_features):
                    match_or_not = False

                    # match ternary
                    LPM_table   = table_lpm['feature ' + str(f)]
                    masks       = []
                    action      = []

                    keys        = sorted_keys[f]
                    input_val   = input_feat_val[f]

                    # For each value in LPM table, check if it matches that separation key
                    for cnt in np.sort(keys):
                        # if there is a ternary match
                        if input_val & LPM_table[cnt][0] == LPM_table[cnt][0] & LPM_table[cnt][1]:
                            masks.append(LPM_table[cnt][0])
                            action.append(LPM_table[cnt][2])

                    if masks:
                        max_mask            = max(masks)
                        max_idx             = masks.index(max_mask)
                        lpm_code_list[f]    = action[max_idx][tree]

                    # match exact
                    code_list[f] = table_exact['feature ' + str(f)][str(input_feat_val[f])][tree]

                if not np.array_equal(code_list, lpm_code_list):
                    print('error in exact to ternary match', code_list, lpm_code_list)

                match_or_not    = False
                all_True        = True

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
                    vote_list[tree] = default_vote

            switch_prediction = default_pred

            for key in table_exact['decision']:
                decision_entry  = table_exact["decision"][key]
                all_match       = True
                anomaly_cnt     = 0

                for tree_v in range(num_trees):
                    expected_vote = decision_entry['t' + str(tree_v) + ' vote']
                    if vote_list[tree_v] != expected_vote:
                        all_match       = False
                        anomaly_cnt    += 1
                        if anomaly_cnt > 0:
                            break

                if all_match:
                    switch_prediction = decision_entry['class']
                    match_or_not = True
                    break

            if not match_or_not:
                switch_prediction = default_pred

            switch_test_y_proba += [anomaly_cnt / num_trees]
            switch_test_y       += [switch_prediction]

        return [switch_test_y, switch_test_y_proba]
