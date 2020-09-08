import numpy as np


class DefaultMonitor:
    def __init__(self, metrics, loss_function=None):
        """
        Creates an instance of the default monitor for training.

        :param metrics: Dict of metric names and metric functions
        :param loss_function: Loss function for training
        """
        self.metrics = metrics
        self.metric_values_train = {metric_name: [] for metric_name in metrics.keys()}
        self.metric_values_val = {metric_name: [] for metric_name in metrics.keys()}
        self.loss_function = loss_function
        self.loss_values = []

    def add_train_result(self, pred, y, loss_value):
        """
        This method is called for each training batch.

        :param pred: Predicted data
        :param y: Ground truth data
        :param loss_value: Loss value from the loss function
        :return:
        """
        for metric_name, metric in self.metrics.items():
            self.metric_values_train[metric_name].append(metric(pred, y))
        self.loss_values.append(loss_value)

    def add_val_result(self, pred, y):
        """
        This method is called for each validation batch.

        :param pred: Predicted data
        :param y: Ground truth data
        """
        for metric_name, metric in self.metrics.items():
            self.metric_values_val[metric_name].append(metric(pred, y))

    def get_train_summary(self):
        """
        Returns a summary of the achieved training results.

        :return: A string with the training performance
        """
        metric_strings = ["{} = {:0.4f}".format(item[0], np.mean(item[1])) for item in
                          self.metric_values_train.items()]
        ret = ",".join(metric_strings)
        ret += ", {} = {:0.4f}".format(self.loss_function.__name__, np.mean(self.loss_values))
        return ret

    def get_val_summary(self):
        """
        Returns a summary of the achieved validation results.

        :return: A string with the validation performance
        """
        metric_strings = ["{} = {:0.4f}".format(item[0], np.mean(item[1])) for item in
                          self.metric_values_val.items()]
        ret = ",".join(metric_strings)
        val_scores = self.get_val_score()
        val_score = np.mean(list(val_scores.values()))
        ret += ", val_score = {:0.4f}".format(val_score)
        return ret

    def get_val_score(self):
        """
        Returns a score describing the performance on the validation set.

        :return: Dict with metric names and average scores for that metrics
        """
        return {item[0]: np.mean(item[1]) for item in
                self.metric_values_val.items()}

    def reset(self):
        """
        This method is called after each epoch in order to reset the results.

        """
        self.metric_values_train = {metric_name: [] for metric_name in self.metrics.keys()}
        self.metric_values_val = {metric_name: [] for metric_name in self.metrics.keys()}
        self.loss_values = []

    def is_better_than(self, best_val_result):
        """
        Compares the provided result (from a preceding epoch) to the achieved results in the current epoch.

        :param best_val_result: Validation result to which the current results are to be compared
        :return: True, if the current result is better than the provided result, False otherwise
        """

        if best_val_result is None:
            return True
        val_scores = self.get_val_score()
        # Average over all metrics for comparison, higher is better
        if np.mean(list(val_scores.values())) > np.mean(list(best_val_result.values())):
            return True
        else:
            return False
