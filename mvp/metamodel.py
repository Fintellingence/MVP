import os
import logging
import logging.handlers
from datetime import datetime as dt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
from tqdm import tqdm
import tensorflow.summary as summary
from tensorboard.plugins.hparams import api as hp
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    precision_score,
    auc,
)

from mvp.selection import (
    count_occurrences,
    avg_uniqueness,
    time_weights,
    sample_weights,
    raw_horizon,
    PEKFold,
)


# TODO: Consider the use of HParams objects insted of dicts
def save_hp_performance(
    base_dir,
    kwargs_model,
    kwargs_scaler,
    avg_metrics,
    hist_data,
    metric_names,
    step,
):
    log_dir = os.path.join(base_dir, "hp_set-" + str(step))
    with summary.create_file_writer(log_dir).as_default():
        hparams = {}
        hparams.update(kwargs_model)
        hparams.update(kwargs_scaler)
        hp.hparams(hparams)
        for name, value in avg_metrics.items():
            summary.scalar(name, value, step=step)
        for name, value in zip(metric_names, hist_data):
            summary.histogram(name, value, step=step)


def create_log_file(name, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        "logs/{}.log".format(name), maxBytes=200 * 1024 * 1024, backupCount=1
    )
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s \n\t%(message)s"
        )
    )
    logger = logging.getLogger("Logger")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    return logger


class BagingModelOptimizer:
    def __init__(
        self,
        model_fn,
        run_dir,
        scaler_pipeline=None,
        cv_splits_fit=10,
        cv_splits_hp=3,
        use_weight=True,
        num_of_threads=None,
        min_time_weight=0.25,
        time_weights_fn=time_weights,
        samples_weights_fn=sample_weights,
        log_dir="logs",
        metrics={
            "Precision": (precision_score, False),
            "F1": (f1_score, False),
            "BAcc": (balanced_accuracy_score, False),
            # "AUC": (auc, False),
        },
    ):
        self._log_dir = log_dir
        self._metrics = metrics
        self._run_dir = run_dir
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
        self._best_kwargs_model = None
        self._best_kwargs_scaler = None
        self._file_log = create_log_file("metamodel", log_dir=self._log_dir)

    def fit(
        self,
        data,
        labels,
        horizon,
        kwargs_model=None,
        kwargs_scaler=None,
        validation=True,
        return_roc=True,
        return_partial_labels=True,
    ):
        pass

    def __partial_fit_eval(
        self,
        data,
        labels,
        train_idx,
        test_idx,
        horizon,
        np_horizon,
        closed,
        kwargs_model,
        kwargs_scaler=None,
    ):
        if kwargs_scaler is None:
            kwargs_scaler = {}
        if self._scaler_pipeline is not None:
            scaler = self._scaler_pipeline.set_params(**kwargs_scaler)
        else:
            scaler = None
        if self._use_weight:
            train_weight = self.get_weight(closed, horizon.iloc[train_idx])
            test_weight = self.get_weight(closed, horizon.iloc[test_idx])
        else:
            train_weight = None
            test_weight = None
        model = self._model_fn(**kwargs_model)

        x_train, x_test = data[train_idx], data[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        train_horizon = np_horizon[train_idx]
        if scaler is not None:
            x_train = scaler.fit_transform(x_train)
        if scaler is not None:
            x_test = scaler.transform(x_test)

        model.fit(
            x_train,
            y_train,
            sample_weight=train_weight,
            horizon=train_horizon,
        )
        predicted = model.predict(x_test)

        outcomes = {}
        for name, (metric_fn, use_weight) in self._metrics.items():
            if use_weight:
                outcomes[name] = metric_fn(
                    y_test, predicted, sample_weight=test_weight
                )
            else:
                outcomes[name] = metric_fn(y_test, predicted)
        return outcomes, model, predicted

    def __is_better(self, last, best):
        for l, b in zip(last, best):
            if l > b:
                return True
        return False

    def hp_meta_optimize(
        self, data, labels, horizon, closed, hp_model, hp_scaler=None
    ):
        if self._use_weight:
            if (self._time_weights_fn is None) and (
                self._samples_weights_fn is None
            ):
                raise ValueError(
                    "With `use_weight` is True, at least one of the functions,"
                    " `time_weight_fn` or `sample_weights_fn`, have to be defined"
                )
        if hp_scaler is None:
            hp_scaler = {}
        grid_model = ParameterGrid(hp_model)
        grid_scaler = ParameterGrid(hp_scaler)
        np_horizon = raw_horizon(closed.index, horizon)
        base_dir = os.path.join(self._run_dir, "hp_tuning")

        n_steps = 1
        for v in hp_model.values():
            n_steps *= len(v)
        for v in hp_scaler.values():
            n_steps *= len(v)
        hp_bar = tqdm(total=n_steps, desc="HP Grid Search with CV")

        hp_step = 0
        metric_names = list(self._metrics.keys())
        best_metric_value = np.zeros(len(self._metrics))
        pekfold = PEKFold(n_splits=self._cv_splits_hp)
        for kwargs_model in grid_model:
            for kwargs_scaler in grid_scaler:
                cv_step = 0
                cv_bar = tqdm(
                    total=self._cv_splits_hp,
                    desc="KFold CV",
                    leave=False,
                )
                cv_hist_values = np.zeros(
                    (self._cv_splits_hp, len(metric_names))
                )
                cv_metric_values = np.zeros(len(self._metrics))
                spliter = pekfold.split(horizon)
                for train_idx, test_idx in spliter:
                    outcomes, _, _ = self.__partial_fit_eval(
                        data,
                        labels,
                        train_idx,
                        test_idx,
                        horizon,
                        np_horizon,
                        closed,
                        kwargs_model,
                        kwargs_scaler,
                    )
                    partial_metric_value = np.array(list(outcomes.values()))
                    cv_hist_values[cv_step] = partial_metric_value
                    cv_metric_values += partial_metric_value
                    cv_step += 1
                    cv_bar.update()
                    cv_bar.set_postfix(**outcomes)
                cv_bar.close()
                cv_metric_values /= self._cv_splits_hp
                if self.__is_better(cv_metric_values, best_metric_value):
                    best_metric_value = cv_metric_values
                    self._best_kwargs_model = kwargs_model
                    self._best_kwargs_scaler = kwargs_scaler
                cv_metric_values = dict(
                    zip(
                        ["avg_" + name for name in metric_names],
                        cv_metric_values,
                    )
                )
                hp_bar.update()
                save_hp_performance(
                    base_dir,
                    kwargs_model,
                    kwargs_scaler,
                    cv_metric_values,
                    cv_hist_values.T,
                    metric_names,
                    hp_step,
                )
                hp_step += 1

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
        # TODO: Possible error during the weight computation when the first event happens in the first timestamp
        sw = sample_weights(
            occurrences,
            horizon,
            closed,
            num_of_threads=self._num_of_threads,
            chunck_size=chunk_size,
        )
        weights = (tw * sw).values
        return weights


class EnvironmentOptimizer(BagingModelOptimizer):
    def __init__(
        self,
        model_fn,
        primary_model_fn,
        labels_fn,
        refined_data,
        author,
        cv_splits_fit=10,
        cv_splits_hp=3,
        use_weight=True,
        num_of_threads=None,
        scaler_pipeline=None,
        min_time_weight=0.25,
        time_weights_fn=time_weights,
        samples_weights_fn=sample_weights,
        log_dir="logs",
        metrics={
            "Precision": (precision_score, False),
            "F1": (f1_score, False),
            "BAcc": (balanced_accuracy_score, False),
            # "AUC": (auc, False),
        },
    ):
        self._log_dir = log_dir
        self._labels_fn = labels_fn
        self._primary_model_fn = primary_model_fn
        self._refined_data = refined_data
        self._cv_splits_hp = cv_splits_hp
        self._file_log = create_log_file("environment", log_dir=self._log_dir)
        self._run_dir = os.path.join(
            self._log_dir, author, "run-" + dt.now().strftime("%Y%m%d-%H%M%S")
        )
        super(EnvironmentOptimizer, self).__init__(
            model_fn,
            self._run_dir,
            scaler_pipeline,
            cv_splits_fit,
            cv_splits_hp,
            use_weight,
            num_of_threads,
            min_time_weight,
            time_weights_fn,
            samples_weights_fn,
            log_dir,
            metrics,
        )

    # TODO: Implement case 2
    def __set_env(
        self, kwargs_features, kwargs_primary, kwargs_labels, approach=1
    ):
        steps = ["[Labels]"]
        for feature_name in kwargs_features.keys():
            steps.append("[" + feature_name + "]")
        bar = tqdm(total=len(steps), desc="Setting Environment [Primary]")
        steps = iter(steps)

        # TODO: Implement a way to process different parameter (volume, frac diff) and a better way to access the raw data
        closed = self._refined_data.df.Close
        primary_model = self._primary_model_fn(
            self._refined_data, **kwargs_primary
        )

        bar.update()
        bar.set_description("Setting Environment {}".format(next(steps)))

        labels_obj = self._labels_fn(
            primary_model.events, closed, **kwargs_labels
        )
        label_data = labels_obj.label_data
        zero_mask = label_data.Label != 0
        label_data = label_data.loc[zero_mask, :]
        horizon = label_data.PositionEnd.copy()
        sides = label_data.Side.copy()
        labels = label_data.Label.copy()
        labels.replace(-1, 0, inplace=True)

        # TODO: Discuss other solutions, instead of drop events
        stats = []
        event_index = horizon.index
        for get_stat_name, kwargs in kwargs_features.items():
            bar.update()
            bar.set_description("Setting Environment {}".format(next(steps)))
            feature = self._refined_data.__getattribute__(get_stat_name)(
                **kwargs
            )
            feature_index = feature.index
            if feature_index[0] <= event_index[0]:
                feature = feature.loc[event_index]
            else:
                mask = (event_index >= feature_index[0]).values
                horizon = horizon.loc[mask]
                sides = sides.loc[mask]
                labels = labels.loc[mask]
                self._file_log.warning(
                    "{} events were droped due to {} with {}"
                    " starts before the first event".format(
                        (~mask).sum(), get_stat_name, kwargs
                    )
                )
            stats.append(feature.values)
        stats.append(sides.values)
        data = np.stack(stats, axis=1)

        self._file_log.info(
            "{} of {} events with label ZERO were droped".format(
                (~zero_mask).sum(), zero_mask.shape[0]
            )
        )
        self._file_log.info(
            "The environment was setted with {} prices and {} events"
            " such that {} are labeled as 1 and {} as -1".format(
                closed.shape[0],
                horizon.shape[0],
                (labels == 1).sum(),
                (labels == 0).sum(),
            )
        )
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
