#!/usr/bin/env python3

import os
import glob
import argparse
import matplotlib.pyplot as plt
import numpy as np
import re

numbers = re.compile(r'(\d+)')

dict_size               = {'2': 'L', '3': 'H'}
dict_metrics_plot       = {'Accuracy:'  : 'Accuracy',
                           'Precision:' : 'Precision',
                           'Recall:'    : 'Recall',
                           'F1'         : 'F1-Score',
                           'AuC:'       : 'AUC'}
dict_metrics_filename   = {'Accuracy:'  : 'acc',
                           'Precision:' : 'precision',
                           'Recall:'    : 'recall',
                           'F1'         : 'f1',
                           'AUC'        : 'auc'}

metrics_match   = ['Accuracy:', 'Precision:', 'Recall:', 'F1', 'AuC:']
metrics_plot    = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']

cur_metrics     = []

SCRIPT_DIR  = os.path.dirname(os.path.realpath(__file__))

BLUE    = '#2171B5'
GREEN   = '#74C476'

def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def get_data_model(cur_dataset, cur_model):
    METRIC_DIR = f'{SCRIPT_DIR}/../metrics/{cur_dataset}/{cur_model}'

    data_files_pattern = f'{METRIC_DIR}/*sklearn*.txt'
    data_files = sorted(glob.glob(data_files_pattern), key=numericalSort)

    if not data_files:
        data_files_pattern = f'{METRIC_DIR}/*pytorch*.txt'
        data_files = sorted(glob.glob(data_files_pattern), key=numericalSort)

    data = {}

    for data_file in data_files:
        file_name = os.path.basename(data_file)
        base_name = os.path.splitext(file_name)[0]

        model       = base_name.rstrip().split('-')[-10]
        model_size  = base_name.rstrip().split('-')[-9]
        trace      = re.search(rf'^(.*?)(?=-{re.escape(model)})', base_name).group(1)

        try:
            model_size = dict_size[model_size]
        except KeyError:
            continue

        for metric in metrics_match:
            with open(data_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.rstrip().split(' ')

                    if line[0] != metric:
                        continue

                    if trace not in data:
                        data[trace] ={}

                    if model_size not in data[trace]:
                        data[trace][model_size] = {}

                    if metric not in data[trace][model_size]:
                        data[trace][model_size][metric] = []

                    data[trace][model_size][metric].append(float(line[-1]))

    stats = []
    for trace in data.keys():
        for model_size in data[trace].keys():
            for metric in data[trace][model_size].keys():
                val = data[trace][model_size][metric]
                stats.append((trace, model_size, metric, val))

    print(stats)
    return stats

def get_data_switch(cur_dataset, cur_model):
    METRIC_DIR = f'{SCRIPT_DIR}/../metrics/{cur_dataset}/{cur_model}'

    data_files_pattern = f'{METRIC_DIR}/*switch*.txt'
    data_files = sorted(glob.glob(data_files_pattern), key=numericalSort)

    data = {}

    for data_file in data_files:
        file_name = os.path.basename(data_file)
        base_name = os.path.splitext(file_name)[0]

        model       = base_name.rstrip().split('-')[-10]
        model_size  = base_name.rstrip().split('-')[-9]
        trace      = re.search(rf'^(.*?)(?=-{re.escape(model)})', base_name).group(1)

        try:
            model_size = dict_size[model_size]
        except KeyError:
            continue


        for metric in metrics_match:
            with open(data_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.rstrip().split(' ')

                    if line[0] != metric:
                        continue

                    if trace not in data:
                        data[trace] ={}

                    if model_size not in data[trace]:
                        data[trace][model_size] ={}

                    if metric not in data[trace][model_size]:
                        data[trace][model_size][metric] = []

                    data[trace][model_size][metric].append(float(line[-1]))

                    if metric not in cur_metrics:
                        cur_metrics.append(metric)

    stats = []
    for trace in data.keys():
        for model_size in data[trace].keys():
            for metric in data[trace][model_size].keys():
                val = data[trace][model_size][metric]
                stats.append((trace, model_size, metric, val))

    print(stats)
    return stats

def gen_plot(data_model, data_switch, cur_dataset, cur_model):
    for metric in cur_metrics:
        # extract all data for the current metric
        cur_data_model   = []
        cur_data_switch  = []
        for d in data_model:
            if metric in d:
                cur_data_model.append(d)
        for d in data_switch:
            if metric in d:
                cur_data_switch.append(d)

        # Obtain the current number of traces to be plotted
        num_traces = 0
        cur_traces = []
        for d in cur_data_switch:
            if d[0] not in cur_traces:
                cur_traces.append(d[0])
                num_traces += 1

        fig, ax = plt.subplots(1, num_traces, sharey=True)
        fig.set_size_inches(15, 3)

        alphas_m = ([.3, .3, .3, .3])
        alphas_s = ([.3, .3, .3, .3])

        rgba_blue       = np.zeros((4,4))
        rgba_blue[:,0]  = 0.12941176470588237
        rgba_blue[:,1]  = 0.44313725490196076
        rgba_blue[:,2]  = 0.7098039215686275
        rgba_blue[:,3]  = alphas_m

        rgba_green      = np.zeros((4,4))
        rgba_green[:,0] = 0.4549019607843137
        rgba_green[:,1] = 0.7686274509803922
        rgba_green[:,2] = 0.4627450980392157
        rgba_green[:,3] = alphas_s

        trace               = [ d for d in cur_traces ]
        model_size          = [ d[1] for d in cur_data_switch ]
        metric_switch       = [ float(d[3][0]) for d in cur_data_switch ]
        metric_model        = [ float(d[3][0]) for d in cur_data_model ]
        metric_switch_high  = [ x - 0.5 if x > 0.5 else 0 for x in metric_switch ]
        metric_switch_low   = [ x if x <= 0.5 else 0.5 for x in metric_switch ]
        metric_model_high   = [ x - 0.5 if x > 0.5 else 0 for x in metric_model ]
        metric_model_low    = [ x if x <= 0.5 else 0.5 for x in metric_model ]

        model_size = model_size[0:2]

        for cur_trace in range(num_traces):
            ax[cur_trace].bar(model_size, metric_model_high[2*cur_trace:2*cur_trace+2],
                              bottom=0.5, color=BLUE, align='edge', width=-0.4,
                              label='Model', hatch='\\\\')
            ax[cur_trace].bar(model_size, metric_model_low[2*cur_trace:2*cur_trace+2], 
                              color=rgba_blue, align='edge', width=-0.4,
                              label='Model', hatch='\\\\', edgecolor=rgba_blue, linewidth=0)
            ax[cur_trace].bar(model_size, metric_switch_high[2*cur_trace:2*cur_trace+2],
                              bottom=0.5, color=GREEN, align='edge', width=0.4,
                              label='Switch', hatch='/')
            ax[cur_trace].bar(model_size, metric_switch_low[2*cur_trace:2*cur_trace+2], 
                              color=rgba_green, align='edge', width=0.4,
                              label='Switch', hatch='/', edgecolor=rgba_green, linewidth=0)
            ax[cur_trace].set_xlabel(trace[cur_trace])
            ax[cur_trace].tick_params(axis='x')
            # if metric == 'AuC:':
            ax[cur_trace].axhline(0.5, color='darkslategrey', lw=0.5, linestyle='dashed')
            # else:
            #     ax[cur_trace].set_yscale('log')
                # ax[cur_trace].minorticks_off()

            if cur_trace > 0:
                plt.setp(ax[cur_trace].get_yticklabels(), visible=False)


        ax[0].set_ylabel(dict_metrics_plot[metric])
        # if metric == 'AuC:':
        ax[0].set_yticks( [ 0, 0.2, 0.4, 0.6, 0.8, 1 ], ['0', 0.2, 0.4, 0.6, 0.8, '1'])
        # else:
            # ax[0].set_yticks( [ 0, 0.001, 0.01, 0.1, 1 ])
            # ax[0].set_ylim(0.0001,1)

        handles, labels = ax[0].get_legend_handles_labels()
        order = [0,2]
        fig.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc='lower center',bbox_to_anchor=(0.5, -0.185),ncol=2)

        # ax[0].bar(model_size, metric_model_high[0:3], bottom=0.5, color=BLUE, align='edge',
        #         width=-0.4, label='Model', hatch='\\\\')
        # ax[0].bar(model_size, metric_model_low[0:3], color=rgba_blue, align='edge', width=-0.4,
        #         label='Model', hatch='\\\\', edgecolor=rgba_blue, linewidth=0)
        # ax[0].bar(model_size, metric_switch_high[0:3], bottom=0.5, color=GREEN, align='edge',
        #         width=0.4, label='Switch', hatch='/')
        # ax[0].bar(model_size, metric_switch_low[0:3], color=rgba_green, align='edge',
        #         width=0.4, label='Switch', hatch='/', edgecolor=rgba_green, linewidth=0)
        # ax[0].set_ylabel('acc')
        # ax[0].set_xlabel(trace[0])
        # ax[0].tick_params(axis='x')
        # ax[0].set_yticks( [ 0, 0.2, 0.4, 0.6, 0.8, 1 ], ['0', 0.2, 0.4, 0.6, 0.8, '1'])
        # ax[0].axhline(0.5, color='darkslategrey', lw=0.5, linestyle='dashed')

        # handles, labels = ax[0].get_legend_handles_labels()
        # order = [0,2]
        # fig.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc='lower center',bbox_to_anchor=(0.5, -0.185),ncol=2)

        # ax[1].bar(model_size, metric_model_high[3:6], bottom=0.5, color=BLUE, align='edge',
        #         width=-0.4, label='Model', hatch='\\\\')
        # ax[1].bar(model_size, metric_model_low[3:6], color=rgba_blue, align='edge', width=-0.4,
        #         label='Model', hatch='\\\\', edgecolor=rgba_blue, linewidth=0)
        # ax[1].bar(model_size, metric_switch_high[3:6], bottom=0.5, color=GREEN, align='edge',
        #         width=0.4, label='Switch', hatch='/')
        # ax[1].bar(model_size, metric_switch_low[3:6], color=rgba_green, align='edge',
        #         width=0.4, label='Switch', hatch='/', edgecolor=rgba_green, linewidth=0)
        # ax[1].set_xlabel(trace[3])
        # ax[1].tick_params(axis='x')
        # ax[1].axhline(0.5, color='darkslategrey', lw=0.5, linestyle='dashed')
        # plt.setp(ax[1].get_yticklabels(), visible=False)

        # ax[2].bar(model_size, metric_model_high[6:9], bottom=0.5, color=BLUE, align='edge',
        #         width=-0.4, label='Model', hatch='\\\\')
        # ax[2].bar(model_size, metric_model_low[6:9], color=rgba_blue, align='edge', width=-0.4,
        #         label='Model', hatch='\\\\', edgecolor=rgba_blue, linewidth=0)
        # ax[2].bar(model_size, metric_switch_high[6:9], bottom=0.5, color=GREEN, align='edge',
        #         width=0.4, label='Switch', hatch='/')
        # ax[2].bar(model_size, metric_switch_low[6:9], color=rgba_green, align='edge',
        #         width=0.4, label='Switch', hatch='/', edgecolor=rgba_green, linewidth=0)
        # ax[2].set_xlabel(trace[6])
        # ax[2].tick_params(axis='x')
        # ax[2].axhline(0.5, color='darkslategrey', lw=0.5, linestyle='dashed')
        # plt.setp(ax[2].get_yticklabels(), visible=False)

        # ax[3].bar(model_size, metric_model_high[9:12], bottom=0.5, color=BLUE, align='edge',
        #         width=-0.4, label='Model', hatch='\\\\')
        # ax[3].bar(model_size, metric_model_low[9:12], color=rgba_blue, align='edge', width=-0.4,
        #         label='Model', hatch='\\\\', edgecolor=rgba_blue, linewidth=0)
        # ax[3].bar(model_size, metric_switch_high[9:12], bottom=0.5, color=GREEN, align='edge',
        #         width=0.4, label='Switch', hatch='/')
        # ax[3].bar(model_size, metric_switch_low[9:12], color=rgba_green, align='edge',
        #         width=0.4, label='Switch', hatch='/', edgecolor=rgba_green, linewidth=0)
        # ax[3].set_xlabel(trace[9])
        # ax[3].tick_params(axis='x')
        # ax[3].axhline(0.5, color='darkslategrey', lw=0.5, linestyle='dashed')
        # plt.setp(ax[3].get_yticklabels(), visible=False)

        # ax[4].bar(model_size, metric_model_high[12:15], bottom=0.5, color=BLUE, align='edge',
        #         width=-0.4, label='Model', hatch='\\\\')
        # ax[4].bar(model_size, metric_model_low[12:15], color=rgba_blue, align='edge', width=-0.4,
        #         label='Model', hatch='\\\\', edgecolor=rgba_blue, linewidth=0)
        # ax[4].bar(model_size, metric_switch_high[12:15], bottom=0.5, color=GREEN, align='edge',
        #         width=0.4, label='Switch', hatch='/')
        # ax[4].bar(model_size, metric_switch_low[12:15], color=rgba_green, align='edge',
        #         width=0.4, label='Switch', hatch='/', edgecolor=rgba_green, linewidth=0)
        # ax[4].set_xlabel(trace[12])
        # ax[4].tick_params(axis='x')
        # ax[4].axhline(0.5, color='darkslategrey', lw=0.5, linestyle='dashed')
        # plt.setp(ax[4].get_yticklabels(), visible=False)

        # ax[5].bar(model_size, metric_model_high[15:18], bottom=0.5, color=BLUE, align='edge',
        #         width=-0.4, label='Model', hatch='\\\\')
        # ax[5].bar(model_size, metric_model_low[15:18], color=rgba_blue, align='edge', width=-0.4,
        #         label='Model', hatch='\\\\', edgecolor=rgba_blue, linewidth=0)
        # ax[5].bar(model_size, metric_switch_high[15:18], bottom=0.5, color=GREEN, align='edge',
        #         width=0.4, label='Switch', hatch='/')
        # ax[5].bar(model_size, metric_switch_low[15:18], color=rgba_green, align='edge',
        #         width=0.4, label='Switch', hatch='/', edgecolor=rgba_green, linewidth=0)
        # ax[5].set_xlabel(trace[15])
        # ax[5].tick_params(axis='x')
        # ax[5].axhline(0.5, color='darkslategrey', lw=0.5, linestyle='dashed')
        # plt.setp(ax[5].get_yticklabels(), visible=False)

        # ax[6].bar(model_size, metric_model_high[18:21], bottom=0.5, color=BLUE, align='edge',
        #         width=-0.4, label='Model', hatch='\\\\')
        # ax[6].bar(model_size, metric_model_low[18:21], color=rgba_blue, align='edge', width=-0.4,
        #         label='Model', hatch='\\\\', edgecolor=rgba_blue, linewidth=0)
        # ax[6].bar(model_size, metric_switch_high[18:21], bottom=0.5, color=GREEN, align='edge',
        #         width=0.4, label='Switch', hatch='/')
        # ax[6].bar(model_size, metric_switch_low[18:21], color=rgba_green, align='edge',
        #         width=0.4, label='Switch', hatch='/', edgecolor=rgba_green, linewidth=0)
        # ax[6].set_xlabel(trace[18])
        # ax[6].tick_params(axis='x')
        # ax[6].axhline(0.5, color='darkslategrey', lw=0.5, linestyle='dashed')
        # plt.setp(ax[6].get_yticklabels(), visible=False)

        # ax[7].bar(model_size, metric_model_high[21:24], bottom=0.5, color=BLUE, align='edge',
        #         width=-0.4, label='Model', hatch='\\\\')
        # ax[7].bar(model_size, metric_model_low[21:24], color=rgba_blue, align='edge', width=-0.4,
        #         label='Model', hatch='\\\\', edgecolor=rgba_blue, linewidth=0)
        # ax[7].bar(model_size, metric_switch_high[21:24], bottom=0.5, color=GREEN, align='edge',
        #         width=0.4, label='Switch', hatch='/')
        # ax[7].bar(model_size, metric_switch_low[21:24], color=rgba_green, align='edge',
        #         width=0.4, label='Switch', hatch='/', edgecolor=rgba_green, linewidth=0)
        # ax[7].set_xlabel(trace[21])
        # ax[7].tick_params(axis='x')
        # ax[7].axhline(0.5, color='darkslategrey', lw=0.5, linestyle='dashed')
        # plt.setp(ax[7].get_yticklabels(), visible=False)

        # if metric == 'AuC:':
        plt.ylim(0, 1)

        PLOT = f'{SCRIPT_DIR}/{cur_model}-{cur_dataset}-{dict_metrics_filename[metric]}.png'

        plt.savefig(PLOT, dpi=500, format="png", bbox_inches="tight")

def plot(cur_dataset, cur_model):
    data_model  = get_data_model(cur_dataset, cur_model)
    data_switch = get_data_switch(cur_dataset, cur_model)
    gen_plot(data_model, data_switch, cur_dataset, cur_model)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="peregrine-py")
    argparser.add_argument('--dataset', type=str, help='Current dataset dir name.')
    argparser.add_argument('--model', type=str, help='Current model.')
    args = argparser.parse_args()

    plot(args.dataset, args.model)
