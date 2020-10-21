from typing import Optional


class TimeSeriesSplit:
    """Split data for nested (rolling) time-series cross-validation"""

    def __init__(self,
                 test_period_len: int,
                 train_period_len: Optional[int] = None,
                 n_predictions: int = 1,
                 prediction_lag: int = 0,
                 prediction_step: int = 1,
                 leave_n_first_points: int = 0,
                 leave_n_last_points: int = 0):
        """
        :param test_period_len: Total number of points in the test period
        :param train_period_len: Maximum number of points in the train period
            *None* means all points will be used
        :param n_predictions: Predict <x> points ahead
            E.g., *1* means that 1 value will be forecasted in every CV fold
        :param prediction_step: Place CV folds every <x> points
            A value greater than 1 will help to speed up CV
        :param prediction_lag: Lag predictions by <x> points
            A value greater than 0 means that there is a gap between the train and test period
            Must be zero for autoregressive models
        """

        assert test_period_len > 0
        assert (train_period_len is None) or (test_period_len > 0)
        assert n_predictions >= 1
        assert prediction_lag >= 0
        assert leave_n_first_points >= 0
        assert leave_n_last_points >= 0

        self.test_period_len = test_period_len
        self.train_period_len = train_period_len
        self.n_predictions = n_predictions
        self.prediction_step = prediction_step
        self.prediction_lag = prediction_lag
        self.leave_n_first_points = leave_n_first_points
        self.leave_n_last_points = leave_n_last_points

    @staticmethod
    def _cv_gen(last_idx: int,
                test_period_len: int,
                train_period_len: Optional[int] = None,
                n_predictions: int = 1,
                prediction_lag: int = 0,
                prediction_step: int = 1,
                leave_n_first_points: int = 0,
                leave_n_last_points: int = 0):
        last_end_test_idx = last_idx - leave_n_last_points
        last_end_train_idx = last_end_test_idx - n_predictions - prediction_lag
        first_end_train_idx = last_end_train_idx - test_period_len + prediction_step
        first_end_train_idx = first_end_train_idx + (last_end_train_idx - first_end_train_idx) % prediction_step
        if first_end_train_idx < leave_n_first_points:
            raise AssertionError(f"First training interval is empty: "
                                 f"{first_end_train_idx} < {leave_n_first_points}")

        for end_train_idx in range(first_end_train_idx, last_end_train_idx + 1,
                                   prediction_step):
            if train_period_len is not None:
                start_train_idx = max(end_train_idx - train_period_len + 1, leave_n_first_points)
            else:
                start_train_idx = leave_n_first_points
            train_ids = slice(start_train_idx,
                              min(end_train_idx, last_idx) + 1)
            test_ids = slice(end_train_idx + prediction_lag + 1,
                             min(end_train_idx + prediction_lag + n_predictions, last_idx) + 1)
            yield train_ids, test_ids

    def split(self, X, y=None, groups=None):
        return self._cv_gen(last_idx=len(X) - 1,
                            test_period_len=self.test_period_len,
                            train_period_len=self.train_period_len,
                            n_predictions=self.n_predictions,
                            prediction_step=self.prediction_step,
                            prediction_lag=self.prediction_lag,
                            leave_n_first_points=self.leave_n_first_points,
                            leave_n_last_points=self.leave_n_last_points)

    def get_n_splits(self, X, y=None, groups=None):
        n_splits = self.test_period_len // self.prediction_step
        return n_splits
