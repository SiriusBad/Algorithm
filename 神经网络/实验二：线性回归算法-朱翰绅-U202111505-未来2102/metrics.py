import numpy as np
from collections import Counter

class Metric:
    def __init__(self, y_pred,y_true):
        self.y_pred = y_pred
        self.y_true = y_true


    def accuracy(self):
        correct = np.sum(self.y_true == self.y_pred)
        total = len(self.y_true)
        return correct / total

    def f1_score(self):
        true_positives = np.sum((self.y_true == 1) & (self.y_pred == 1))
        false_positives = np.sum((self.y_true == -1) & (self.y_pred == 1))
        false_negatives = np.sum((self.y_true == 1) & (self.y_pred == -1))

        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)

        f1 = 2 * (precision * recall) / (precision + recall)
        return f1

    def precision(self):
        true_positives = np.sum((self.y_true == 1) & (self.y_pred == 1))
        false_positives = np.sum((self.y_true == -1) & (self.y_pred == 1))

        precision = true_positives / (true_positives + false_positives)
        return precision

    def recall(self):
        true_positives = np.sum((self.y_true == 1) & (self.y_pred == 1))
        false_negatives = np.sum((self.y_true == 1) & (self.y_pred == -1))

        recall = true_positives / (true_positives + false_negatives)
        return recall

    def confusion_matrix(self):
        true_positives = np.sum((self.y_true == 1) & (self.y_pred == 1))
        false_positives = np.sum((self.y_true == -1) & (self.y_pred == 1))
        true_negatives = np.sum((self.y_true == -1) & (self.y_pred == -1))
        false_negatives = np.sum((self.y_true == 1) & (self.y_pred == -1))

        return {
            "True Positives": true_positives,
            "False Positives": false_positives,
            "True Negatives": true_negatives,
            "False Negatives": false_negatives,
        }


