import numpy as np
import pandas as pd
from tf.summary import summary
from tensorboard.plugins.hparams import api as hp

from mvp.labels import Labels
from mvp.selection import (
    count_occurrences,
    avg_uniqueness,
    time_weights,
    sample_weights,
    raw_horizon,
    PEKFold,
)


def report_metrics(predicted, expected):
    pass


class BagingModelOptimizer:
    def __init__(
        self,
        model_fn,
        scaler_pipeline=None,
        cv_splits_fit=10,
        cv_splits_hp=3,
        use_time_weights=True,
        use_samples_weights=True,
    ):
        self._model_fn = model_fn
        self._scaler_pipeline = scaler_pipeline
        self._cv_splits_fit = cv_splits_fit
        self._cv_splits_hp = cv_splits_hp
        self._use_time_weights = use_time_weights
        self._use_samples_weights = use_samples_weights

    def fit(
        self,
        data,
        labels,
        horizon,
        kwargs_model,
        kwargs_scaler=None,
        validation=True,
        return_roc=True,
        return_partial_labels=True,
    ):
        pass

    def partial_fit(
        self,
        data,
        labels,
        horizon,
        kwargs_model,
        kwargs_scaler=None,
    ):
        model = self._model_fn(**kwargs_model)

    def hp_optimize(self, data, labels, horizon, hp_model, hp_scaler=None):
        pass

    def predict(self, data):
        pass


class EnvironmentOptimizer(BagingModelOptimizer):
    def __init__(
        self,
        model_fn,
        primary_model_fn,
        labels_fn,
        refined_data,
        cv_splits_hp=3,
        cv_splits_fit=10,
        scaler_pipeline=None,
        use_time_weights=True,
        use_samples_weights=True,
    ):
        super(EnvironmentOptimizer, self).__init__(
            model_fn,
            scaler_pipeline,
            cv_splits_fit,
            cv_splits_hp,
            use_time_weights,
            use_samples_weights,
        )
        self._labels_fn = labels_fn
        self._primary_model_fn = primary_model_fn
        self._refined_data = refined_data
        self._cv_splits_hp = cv_splits_hp

    def __set_env(self, kwargs_features, kwargs_primary, kwargs_labels):
        # TODO: Implement a way to process different parameter (volume, frac diff) and a better way to access the raw data
        closed = self._refined_data.df.Close
        primary_model = self.primary_model_fn(
            self._refined_data, **kwargs_primary
        )
        labels_obj = self._labels_fn(
            primary_model.events,
            closed,
            operation_parameters=kwargs_labels,
        )
        label_data = labels_obj.label_data
        label_data = label_data.loc[label_data.Label != 0, :]
        horizon = label_data.PositionEnd
        sides = label_data.Sides.values
        labels = label_data.Labels.values
        labels[labels == -1] = 0

        stats = [sides]
        event_index = horizon.index
        for get_stat_name, kwargs in kwargs_features.items():
            stats.append(
                self.__getattribute__(get_stat_name)(**kwargs).loc[event_index]
            )
        data = np.stack(stats, axis=1)
        np_horizon = raw_horizon(closed, horizon)
        return data, labels, np_horizon

    def train(
        self,
        hp_model,
        kwargs_features,
        kwargs_primary,
        kwargs_labels,
        hp_scaler=None,
    ):
        data, labels, horizon = self.__set_env(
            kwargs_features, kwargs_primary, kwargs_labels
        )
        self.hp_optimize(data, labels, horizon, hp_model, hp_scaler)

    def hp_env_optimize(
        self, hp_model, hp_features, hp_primary, hp_labels, hp_scaler=None
    ):
        pass
