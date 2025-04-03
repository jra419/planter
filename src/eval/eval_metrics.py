# import os
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score
from datetime import datetime
from pathlib import Path

ts_datetime = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')[:-3]

def eval_metrics(y_test, y_pred, cur_model, cur_dataset, cur_trace, threshold=-1):
    outdir = f'{Path(__file__).parents[0]}'
    # outpath = os.path.join(outdir, f'{cur_trace}-{cur_model}-{ts_datetime}.csv')

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)

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
