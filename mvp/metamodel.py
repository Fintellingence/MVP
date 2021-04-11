import os

import numpy as np
import pandas as pd
from tf.summary import summary
from tensorboard.plugins.hparams import api as hp
from sklearn.model_selection import ParameterGrid

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
        use_weight=True,
        num_of_threads=None,
        min_time_weight=0.25,
        time_weights_fn=time_weights,
        samples_weights_fn=sample_weights,
    ):
        self._use_weight = use_weight
        self._model_fn = model_fn
        self._scaler_pipeline = scaler_pipeline
        self._cv_splits_fit = cv_splits_fit
        self._cv_splits_hp = cv_splits_hp
        self._min_time_weight = min_time_weight
        self._time_weights_fn = time_weights_fn
        self._samples_weights_fn = samples_weights_fn
        self._num_of_threads = (
            os.cpu_count() if num_of_threads is None else num_of_threads
        )

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

    def partial_fit_eval(
        self,
        data,
        labels,
        train_idx,
        test_idx,
        horizon,
        np_horizon,
        closed,
        metrics,
        kwargs_model,
        kwargs_scaler=None,
    ):
        if kwargs_scaler is None:
            kwargs_scaler = {}
        if self._scaler_pipeline is not None:
            scaler = self._scaler_pipeline(**kwargs_scaler)
        else:
            scaler = None
        model = self._model_fn(**kwargs_model)

        if self._use_weight:
            train_weight = self.get_weight(self, closed, horizon.iloc[train_idx])
            test_weight = self.get_weight(self, closed, horizon.iloc[test_idx])
        else:
            train_weight = None
            test_weight = None
        # TODO: apply scaler and weights
        x_train, x_test = data[train_idx], data[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        train_horizon = np_horizon[train_idx]
        model.fit(
            x_train,
            y_train,
            sample_weight=train_weight,
            horizon=train_horizon,
        )
        predicted = model.predict(x_test)
        outcomes = []
        for metric_fn in metrics:
            outcomes.append(metric_fn(y_test, predicted))

    def hp_meta_optimize(
        self, data, labels, horizon, closed, hp_model, hp_scaler=None
    ):
        if self._use_weight:
            if (self._time_weights_fn is not None) and (
                self._samples_weights_fn is not None
            ):
                raise ValueError(
                    "With `use_weight` is True, at least one of the functions,"
                    " `time_weight_fn` or `sample_weights_fn`, have to be not None"
                )

        grid_model = ParameterGrid(hp_model)
        np_horizon = raw_horizon(closed.index, horizon)
        if hp_scaler is not None:
            grid_scaler = ParameterGrid(hp_model)
        else:
            grid_scaler = [None]

        pekfold = PEKFold()
        for kwargs_model in grid_model:
            for kwargs_scaler in grid_model:
                spliter = pekfold.split(horizon)
                for train_idx, test_idx in spliter:
                    pass

    def get_weight(self, closed, horizon, min_chunk_size=100):
        data_size = horizon.shape[0]
        chunk_size = np.round(data_size / self._num_of_threads)
        if chunk_size < min_chunk_size:
            chunk_size = data_size
        occurrences = count_occurrences(
            closed.index,
            horizon,
            num_of_threads=self._num_of_threads,
            chunck_size=chunk_size,
        )
        avg_uni = avg_uniqueness(
            occurrences,
            horizon,
            num_of_threads=self._num_of_threads,
            chunck_size=chunk_size,
        )
        tw = time_weights(avg_uni, p=self._min_time_weight)
        # TODO: Inpect the behaviour when the first event happens in the first timestamp
        sw = sample_weights(
            occurrences,
            horizon,
            closed,
            num_of_threads=self._num_of_threads,
            chunck_size=chunk_size,
        )
        weights = (tw * sw).values
        return weights

    def predict(self, data):
        pass


class EnvironmentOptimizer(BagingModelOptimizer):
    def __init__(
        self,
        model_fn,
        primary_model_fn,
        labels_fn,
        refined_data,
        cv_splits_fit=10,
        cv_splits_hp=3,
        use_weight=True,
        scaler_pipeline=None,
        min_time_weight=0.25,
        time_weights_fn=time_weights,
        samples_weights_fn=sample_weights,
    ):
        super(EnvironmentOptimizer, self).__init__(
            model_fn,
            scaler_pipeline,
            cv_splits_fit,
            cv_splits_hp,
            use_weight,
            min_time_weight,
            time_weights_fn,
            samples_weights_fn,
        )
        self._labels_fn = labels_fn
        self._primary_model_fn = primary_model_fn
        self._refined_data = refined_data
        self._cv_splits_hp = cv_splits_hp

    # TODO: Implement case 2
    def __set_env(
        self, kwargs_features, kwargs_primary, kwargs_labels, approach=1
    ):
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
        mask = label_data.Label != 0
        label_data = label_data.loc[mask, :]
        print("{} null events were droped".format((~mask).sum()))
        horizon = label_data.PositionEnd
        sides = label_data.Sides
        labels = label_data.Labels
        labels.loc[labels == -1] = 0

        stats = []
        event_index = horizon.index
        # TODO: Discuss other solutions, instead of drop events
        for get_stat_name, kwargs in kwargs_features.items():
            feature = self.__getattribute__(get_stat_name)(**kwargs)
            feature_index = feature.index
            if feature_index[0] <= event_index[0]:
                feature = feature.loc[event_index]
            else:
                mask = (event_index >= feature_index[0]).values
                horizon = horizon.loc[mask]
                sides = sides.loc[mask]
                labels = labels.loc[mask]
                print(
                    "{} events were droped due to {} with {}".format(
                        (~mask).sum(), get_stat_name, kwargs
                    )
                )
            stats.append(feature.values)
        stats.append(sides.values)
        data = np.stack(stats, axis=1)
        return horizon, closed, data, labels.values

    def train(
        self,
        hp_model,
        kwargs_features,
        kwargs_primary,
        kwargs_labels,
        hp_scaler=None,
    ):
        horizon, closed, data, labels = self.__set_env(
            kwargs_features, kwargs_primary, kwargs_labels
        )
        self.hp_meta_optimize(
            data, labels, horizon, closed, hp_model, hp_scaler
        )

    def hp_env_optimize(
        self, hp_model, hp_features, hp_primary, hp_labels, hp_scaler=None
    ):
        pass
