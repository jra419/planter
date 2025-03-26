import copy
import gc
import os
import numpy as np
import pandas as pd
import pickle
import socket
import struct
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def ip2int(addr):
    return struct.unpack("!I", socket.inet_aton(addr))[0]

def ip2long(ip):
    """
    Convert an IP string to long
    """
    packedIP = socket.inet_aton(ip)
    return struct.unpack("!L", packedIP)[0]

def ip2bin(ip):
    ip1 = '.'.join([bin(int(x)+256)[3:] for x in ip.split('.')])
    return ip1

def ip2hex(ip):
    ip1 = '-'.join([hex(int(x)+256)[3:] for x in ip.split('.')])
    return ip1

def bin2dec(ip):
    return int(ip,2)

def load_data(num_features, data_dir):
    label_index = ' label'
    # normal_label = 'BENIGN'

    # file_dir = data_dir+'/CICIDS/'

    # files = []

    df_data    = pd.read_csv(data_dir+'/active-wiretap/active-wiretap.csv',
                             usecols=['ip.src', 'ip.dst', 'ip.proto',
                                      'tcp.srcport', 'tcp.dstport', 'udp.srcport', 'udp.dstport'])

    print('df_data.head before:')
    print(df_data.head())

    df_data['port.src'] = df_data['tcp.srcport'].fillna(df_data['udp.srcport'])
    df_data['port.dst'] = df_data['tcp.dstport'].fillna(df_data['udp.dstport'])

    df_data = df_data.drop('tcp.srcport', axis=1)
    df_data = df_data.drop('tcp.dstport', axis=1)
    df_data = df_data.drop('udp.srcport', axis=1)
    df_data = df_data.drop('udp.dstport', axis=1)

    print('df_data.head after:')
    print(df_data.head())

    df_labels  = pd.read_csv(data_dir+'/active-wiretap/active-wiretap-labels.csv', header=None)

    df_labels.columns = ['label']

    df_full = pd.concat([df_data, df_labels], axis=1)

    print(f'size before: {len(df_full)}')
    df_full = df_full[(df_full['ip.proto'] == 17)
                       | (df_full['ip.proto'] == 6)
                       | (df_full['ip.proto'] == 1)]
    print(f'size after: {len(df_full)}')

    df_full['ip.src'] = df_full['ip.src'].astype(str)
    df_full['ip.dst'] = df_full['ip.dst'].astype(str)
    df_full['ip.src'] = df_full['ip.src'].apply(ip2int)
    df_full['ip.dst'] = df_full['ip.dst'].apply(ip2int)

    # data[['srcip_part_1', 'srcip_part_2', 'srcip_part_3', 'srcip_part_4']] = data[' Source IP'].apply(ip2bin).str.split('.',expand=True)
    # data[['dstip_part_1', 'dstip_part_2', 'dstip_part_3', 'dstip_part_4']] = data[' Destination IP'].apply(ip2bin).str.split('.',expand=True)

    # data['srcip_part_1'] = data['srcip_part_1'].apply(bin2dec)
    # data['srcip_part_2'] = data['srcip_part_2'].apply(bin2dec)
    # data['srcip_part_3'] = data['srcip_part_3'].apply(bin2dec)
    # data['srcip_part_4'] = data['srcip_part_4'].apply(bin2dec)
    # data['dstip_part_1'] = data['dstip_part_1'].apply(bin2dec)
    # data['dstip_part_2'] = data['dstip_part_2'].apply(bin2dec)
    # data['dstip_part_3'] = data['dstip_part_3'].apply(bin2dec)
    # data['dstip_part_4'] = data['dstip_part_4'].apply(bin2dec)

    # data[' Source IP'] = data[' Source IP'].apply(ip2long)
    # data[' Destination IP'] = data[' Destination IP'].apply(ip2long)
    print('df_full.head:')
    print(df_full.head())

    # for key in range(len(data[label_index].values)):
    #     if data[label_index].values[key]=='BENIGN':
    #         data[label_index].values[key] = 0
    #     else:
    #         data[label_index].values[key] = 1



        # percent = np.int(np.ceil(50*key/len(data[label_index].values)))
        # if key%10==0:
        #     print('\rProcessing the raw Data ['+percent*'#'+(50-percent)*'-'+'] '+str(int(np.round(100*key/len(data[label_index].values))))+"%",end="")

    #Replace values with NaN, inf, -inf
    # data.replace([np.inf, -np.inf], np.nan)
    #print('')
    #data.replace([np.inf, -np.inf], np.nan)
    ##Remove rows containing NaN
    # data.dropna(how="any", inplace = True)
    # data = data[data.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1)]

    df_full.describe()
    df_full.info()
    print(df_full['label'].value_counts())

    used_features = ['port.src', 'port.dst', 'ip.proto', 'ip.src', 'ip.dst'][:num_features]

    X = copy.deepcopy(df_full[used_features].astype('int'))
    y = copy.deepcopy(df_full['label'].astype('int'))
    # del data
    # gc.collect()

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=1000000)

    print('dataset is loaded')

    return X_train, np.array(y_train), X_test, np.array(y_test), used_features
