# import os
from sklearn import metrics
from sklearn import metrics
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import torch

ts_datetime = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')[:-3]

def eval_metrics(y_test, y_pred, y_pred_proba, cur_model, cur_dataset, cur_trace, threshold=-1):
    outdir = f'{Path(__file__).parents[0]}'

    tn, fp, fn, tp  = metrics.confusion_matrix(y_test, y_pred).ravel()
    accuracy        = metrics.accuracy_score(y_test, y_pred)
    precision       = metrics.precision_score(y_test, y_pred)
    recall          = metrics.recall_score(y_test, y_pred)
    f1              = metrics.f1_score(y_test, y_pred)
    auc             = metrics.roc_auc_score(y_test, y_pred_proba)

    try:
        tpr = tp / (tp + fn)
    except ZeroDivisionError:
        tpr = 0

    try:
        tnr = tn / (tn + fp)
    except ZeroDivisionError:
        tnr = 0

    try:
        fpr = fp / (fp + tn)
    except ZeroDivisionError:
        fpr = 0

    try:
        fnr = fn / (fn + tp)
    except ZeroDivisionError:
        fnr = 0

    print(f'Accuracy:   {accuracy}')
    print(f'Precision:  {precision}')
    print(f'Recall:     {recall}')
    print(f'F1 Score:   {f1}')
    print(f'AUC:        {auc}')

    # Write the eval to a txt.
    f = open(f'{outdir}/{cur_dataset}-{cur_trace}-{cur_model}-{ts_datetime}.txt', 'a+')
    if threshold != -1:
        f.write(f'Threshold: {threshold}\n')
    f.write(f'TP:           {tp}\n')
    f.write(f'TN:           {tn}\n')
    f.write(f'FP:           {fp}\n')
    f.write(f'FN:           {fn}\n')
    f.write(f'TPR:          {tpr}\n')
    f.write(f'TNR:          {tnr}\n')
    f.write(f'FPR:          {fpr}\n')
    f.write(f'FNR:          {fnr}\n')
    f.write(f'Accuracy:     {accuracy}\n')
    f.write(f'Precision:    {precision}\n')
    f.write(f'Recall:       {recall}\n')
    f.write(f'F1 Score:     {f1}\n')
    f.write(f'AuC:          {auc}\n')

def eval_metrics_ae(labels, mse, cur_model, cur_dataset, cur_trace, threshold=-1):
    # threshold = threshold.detach().numpy()
    # if (threshold == -1):
    #     print(f'Error: invalid threshold value ({threshold}).')
    #     exit()

    outdir = f'{Path(__file__).parents[0]}'

    # Collect the processed packets' RMSE, label.
    df_eval = pd.DataFrame({'mse':mse,'label':labels})
    df_eval = df_eval.astype({'label':'int'})
    print(df_eval.head)

    # Sort by MSE.
    df_eval.sort_values(by='mse', ascending=False, inplace=True)

    # Split by threshold.
    # eval_benign = df_eval[df_eval.mse < threshold]
    # eval_alert  = df_eval[df_eval.mse >= threshold]

    # tp = eval_alert[eval_alert.label == 1].shape[0]
    # fp = eval_alert[eval_alert.label == 0].shape[0]
    # tn = eval_benign[eval_benign.label == 0].shape[0]
    # fn = eval_benign[eval_benign.label == 1].shape[0]

    # try:
    #     tpr = tp / (tp + fn)
    # except ZeroDivisionError:
    #     tpr = 0

    # try:
    #     tnr = tn / (tn + fp)
    # except ZeroDivisionError:
    #     tnr = 0

    # try:
    #     fpr = fp / (fp + tn)
    # except ZeroDivisionError:
    #     fpr = 0

    # try:
    #     fnr = fn / (fn + tp)
    # except ZeroDivisionError:
    #     fnr = 0
    # try:
    #     accuracy = (tp + tn) / (tp + fp + fn + tn)
    # except ZeroDivisionError:
    #     accuracy = 0

    # try:
    #     precision = tp / (tp + fp)
    # except ZeroDivisionError:
    #     precision = 0

    # try:
    #     recall = tp / (tp + fn)
    # except ZeroDivisionError:
    #     recall = 0

    # try:
    #     f1 = 2 * (recall * precision) / (recall + precision)
    # except ZeroDivisionError:
    #     f1 = 0

    roc_curve_fpr, roc_curve_tpr, roc_curve_thres   = metrics.roc_curve(df_eval.label, df_eval.mse)
    roc_curve_fnr                                   = 1 - roc_curve_tpr

    auc         = metrics.roc_auc_score(df_eval.label, df_eval.mse)
    eer         = roc_curve_fpr[np.nanargmin(np.absolute((roc_curve_fnr - roc_curve_fpr)))]
    eer_sanity  = roc_curve_fnr[np.nanargmin(np.absolute((roc_curve_fnr - roc_curve_fpr)))]

    # print(f'Accuracy:   {accuracy}')
    # print(f'Precision:  {precision}')
    # print(f'Recall:     {recall}')
    # print(f'F1 Score:   {f1}')
    # print(f'AUC:        {auc}')

    # Write the eval to a txt.
    f = open(f'{outdir}/{cur_dataset}-{cur_trace}-{cur_model}-{ts_datetime}.txt', 'a+')
    if threshold != -1:
        f.write(f'Threshold: {threshold}\n')
    # f.write(f'TP:           {tp}\n')
    # f.write(f'TN:           {tn}\n')
    # f.write(f'FP:           {fp}\n')
    # f.write(f'FN:           {fn}\n')
    # f.write(f'TPR:          {tpr}\n')
    # f.write(f'TNR:          {tnr}\n')
    # f.write(f'FPR:          {fpr}\n')
    # f.write(f'FNR:          {fnr}\n')
    # f.write(f'Accuracy:     {accuracy}\n')
    # f.write(f'Precision:    {precision}\n')
    # f.write(f'Recall:       {recall}\n')
    # f.write(f'F1 Score:     {f1}\n')
    f.write(f'AUC:          {auc}\n')
