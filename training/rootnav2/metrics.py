# Adapted from score written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import numpy as np

class LocalisationAccuracyMeter():
    def __init__(self, channel_count):
        self.channel_count = channel_count
        self.tp = [0] * channel_count
        self.fp = [0] * channel_count
        self.fn = [0] * channel_count

    def update(self, channel_results):
        for i in range(self.channel_count):
            self.tp[i] += channel_results[i][0]
            self.fp[i] += channel_results[i][1]
            self.fn[i] += channel_results[i][2]

    def f1(self):
        results = []

        for i in range(self.channel_count):
            tp, fp, fn = self.tp[i], self.fp[i], self.fn[i]
            precision, recall = 0.0, 0.0
        
            if tp + fp > 0:
                precision = tp / float(tp + fp)

            if tp + fn > 0:
                recall = tp / float(tp + fn)

            f1 = 0.0
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            results.append((precision, recall, f1))
        return results


class runningScore(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask],
            minlength=n_class ** 2,
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(
                lt.flatten(), lp.flatten(), self.n_classes
            )

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return (
            {
                "oacc": acc,
                "macc": acc_cls,
                "facc": fwavacc,
                "miou": mean_iu,
            },
            cls_iu,
        )

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))



class averageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

