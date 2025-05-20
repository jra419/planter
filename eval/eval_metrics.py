# import os
from sklearn import metrics
from sklearn import metrics
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import torch

ts_datetime = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')[:-3]

def eval_metrics(y_test, y_pred, y_pred_proba, cur_dataset, cur_trace,
                 cur_model, model_size, arch, threshold=-1):
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

    print(f'TP:         {tp}')
    print(f'TN:         {tn}')
    print(f'FP:         {fp}')
    print(f'FN:         {fn}')
    print(f'TPR:        {tpr}')
    print(f'TNR:        {tnr}')
    print(f'FPR:        {fpr}')
    print(f'FNR:        {fnr}')
    print(f'Accuracy:   {accuracy}')
    print(f'Precision:  {precision}')
    print(f'Recall:     {recall}')
    print(f'F1 Score:   {f1}')
    print(f'AUC:        {auc}')

    # Write the eval to a txt.
    f = open(f'{outdir}/metrics/{cur_dataset}/{cur_model}/{cur_trace}-{cur_model}-{model_size}-{arch}-{ts_datetime}.txt', 'a+')
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

def eval_metrics_kmeans(y_test, y_pred, cur_dataset, cur_trace, cur_model,
                        model_size, arch):
    outdir = f'{Path(__file__).parents[0]}'

    tn, fp, fn, tp  = metrics.confusion_matrix(y_test, y_pred).ravel()
    accuracy        = metrics.accuracy_score(y_test, y_pred)
    precision       = metrics.precision_score(y_test, y_pred)
    recall          = metrics.recall_score(y_test, y_pred)
    f1              = metrics.f1_score(y_test, y_pred)

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

    print(f'TP:         {tp}')
    print(f'TN:         {tn}')
    print(f'FP:         {fp}')
    print(f'FN:         {fn}')
    print(f'TPR:        {tpr}')
    print(f'TNR:        {tnr}')
    print(f'FPR:        {fpr}')
    print(f'FNR:        {fnr}')
    print(f'Accuracy:   {accuracy}')
    print(f'Precision:  {precision}')
    print(f'Recall:     {recall}')
    print(f'F1 Score:   {f1}')

    # Write the eval to a txt.
    f = open(f'{outdir}/metrics/{cur_dataset}/{cur_model}/{cur_trace}-{cur_model}-{model_size}-{arch}-{ts_datetime}.txt', 'a+')
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
