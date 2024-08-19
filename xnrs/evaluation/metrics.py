import numpy as np
import sklearn.metrics as metrics


# ranking metrics

from sklearn.metrics import roc_auc_score as auc_score

def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2**y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def false_mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    # NOTE: mrr should actually return the single highest rr not the average...
    return np.sum(rr_score) / np.sum(y_true)


def rr_score(y_true, y_score):
    # correct implementation of rr
    # which averaged over multiple lists to get mrr
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    # returns the single highest rr
    return np.max(rr_score)


def ctr_score(y_true, y_score, k=1):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    return np.mean(y_true)


def acc_score(y_true, y_score):
    y_score = np.round(np.clip(y_score, 0, 1))
    return metrics.accuracy_score(y_true, y_score)


def recall_score(y_true, y_score):
    y_score = np.round(np.clip(y_score, 0, 1))
    return metrics.recall_score(y_true, y_score)


def precision_score(y_true, y_score):
    y_score = np.round(np.clip(y_score, 0, 1))
    return metrics.precision_score(y_true, y_score, zero_division=0)


def confusion_matrix(y_true, y_score):
    y_score = np.round(np.clip(y_score, 0, 1))
    return metrics.confusion_matrix(y_true, y_score)


# regression metrics

from sklearn.metrics import mean_squared_error as mse_score
from sklearn.metrics import mean_absolute_error as mae_score
from sklearn.metrics import r2_score
from scipy.stats import pearsonr as corr_score