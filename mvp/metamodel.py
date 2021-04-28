import os
import logging
import logging.handlers
from joblib import dump
from datetime import datetime as dt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
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

from mvp.refined_data import RefinedData
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
    s_features=None,
    s_primary=None,
    s_labels=None,
):
    log_dir = os.path.join(base_dir, "hp_set-" + str(step))
    with summary.create_file_writer(log_dir).as_default():
        hparams = {}
        hparams.update(kwargs_model)
        hparams.update(kwargs_scaler)
        if s_features is not None:
            hparams["features"] = s_features
        if s_primary is not None:
            hparams["primary"] = s_primary
        if s_labels is not None:
            hparams["labels"] = s_labels
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
        metrics={
            "Precision": (precision_score, False),
            "F1": (f1_score, False),
            "BAcc": (balanced_accuracy_score, False),
            # "AUC": (auc, False),
        },
    ):
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
        self._best_kwargs_scaler = {}
        self._file_log = create_log_file("metamodel", log_dir=self._run_dir)
        if self._use_weight:
            if (self._time_weights_fn is None) and (
                self._samples_weights_fn is None
            ):
                raise ValueError(
                    "With `use_weight` is True, at least one of the functions,"
                    " `time_weight_fn` or `sample_weights_fn`, have to be defined"
                )

    def __is_better(self, last, best):
        for l, b in zip(last, best):
            if l > b:
                return True
        return False

    def __set_model(
        self,
        kwargs_scaler,
        kwargs_model,
        closed,
        train_horizon,
        test_horizon=None,
    ):
        if kwargs_scaler is None:
            kwargs_scaler = {}
        if self._scaler_pipeline is not None:
            scaler = self._scaler_pipeline.set_params(**kwargs_scaler)
        else:
            scaler = None
        if self._use_weight:
            train_weight = self.get_weight(closed, train_horizon)
            if test_horizon is not None:
                test_weight = self.get_weight(closed, test_horizon)
            else:
                test_weight = None
        else:
            train_weight = None
            test_weight = None
        model = self._model_fn(**kwargs_model)
        return model, scaler, train_weight, test_weight

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
        model, scaler, train_weight, test_weight = self.__set_model(
            kwargs_scaler,
            kwargs_model,
            closed,
            horizon[train_idx],
            horizon[test_idx],
        )
        outcomes = {}
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
        for name, (metric_fn, use_weight) in self._metrics.items():
            if use_weight:
                outcomes[name] = metric_fn(
                    y_test, predicted, sample_weight=test_weight
                )
            else:
                outcomes[name] = metric_fn(y_test, predicted)
        return outcomes, predicted, horizon.index[test_idx]

    def __fit(
        self,
        x_train,
        y_train,
        horizon,
        np_horizon,
        closed,
        kwargs_model,
        kwargs_scaler=None,
    ):
        model, scaler, train_weight, _ = self.__set_model(
            kwargs_scaler,
            kwargs_model,
            closed,
            horizon,
        )
        if scaler is not None:
            x_train = scaler.fit_transform(x_train)
        model.fit(
            x_train,
            y_train,
            sample_weight=train_weight,
            horizon=np_horizon,
        )
        return model, scaler

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

    def cv_fit(
        self,
        data,
        labels,
        closed,
        horizon,
        kwargs_model=None,
        kwargs_scaler=None,
        s_features=None,
        s_primary=None,
        s_labels=None,
    ):
        if kwargs_model is None and self._best_kwargs_model is None:
            raise ValueError(
                "At least one of the values, kwargs_model or"
                " _best_kwargs_model, have to be setted"
            )
        elif kwargs_model is None and self._best_kwargs_model is not None:
            kwargs_model = self._best_kwargs_model.copy()
        if kwargs_scaler is None and self._best_kwargs_scaler is None:
            kwargs_scaler = {}
        elif kwargs_scaler is None and self._best_kwargs_scaler is not None:
            kwargs_scaler = self._best_kwargs_scaler.copy()

        metric_names = list(self._metrics.keys())
        np_horizon = raw_horizon(closed.index, horizon)
        base_dir = os.path.join(self._run_dir, "model_fitting")
        pekfold = PEKFold(n_splits=self._cv_splits_fit)
        cv_metric_values = np.zeros(len(self._metrics))
        cv_hist_values = np.zeros((self._cv_splits_fit, len(metric_names)))
        cv_bar = tqdm(
            total=self._cv_splits_fit,
            desc="Assessing the Cross-Validation for the Best Configuration",
            colour="green",
        )

        cv_step = 0
        predictions = []
        spliter = pekfold.split(horizon)
        for train_idx, test_idx in spliter:
            outcomes, predicted, index = self.__partial_fit_eval(
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
            predictions.append(pd.Series(predicted, index=index))
        cv_bar.close()
        cv_metric_values /= self._cv_splits_fit
        cv_metric_values = dict(
            zip(
                ["avg_" + name for name in metric_names],
                cv_metric_values,
            )
        )
        save_hp_performance(
            base_dir,
            kwargs_model,
            kwargs_scaler,
            cv_metric_values,
            cv_hist_values.T,
            metric_names,
            0,
            s_features,
            s_primary,
            s_labels,
        )
        summary_outcomes = (
            "The model fitting and the performance avaliation based on CV were done with"
            "\n\t[Avg Metrics] = {}"
            "\n\t[Features] {}"
            "\n\t[Primary] {}"
            "\n\t[Labels] {}"
            "\n\t[Best model parameters] {}"
            "\n\t[Best scaler parameters] {}".format(
                cv_metric_values,
                s_features,
                s_primary,
                s_labels,
                kwargs_model,
                kwargs_scaler,
            )
        )
        self._file_log.info(summary_outcomes)
        model, scaler = self.__fit(
            data,
            labels,
            horizon,
            np_horizon,
            closed,
            kwargs_model,
            kwargs_scaler,
        )
        dump(model, os.path.join(base_dir, "fitted_model.pkl"))
        if scaler is not None:
            dump(scaler, os.path.join(base_dir, "fitted_scaler.pkl"))
        return model, scaler, predictions, cv_metric_values

    def hp_meta_optimize(
        self,
        data,
        labels,
        horizon,
        closed,
        hp_model,
        hp_scaler=None,
        s_features=None,
        s_primary=None,
        s_labels=None,
    ):
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
        hp_bar = tqdm(
            total=n_steps,
            desc="Grid Searching for Hyper-Parameter",
            colour="green",
        )

        hp_step = 0
        metric_names = list(self._metrics.keys())
        best_metric_value = np.zeros(len(self._metrics))
        pekfold = PEKFold(n_splits=self._cv_splits_hp)
        for kwargs_model in grid_model:
            for kwargs_scaler in grid_scaler:
                cv_step = 0
                cv_bar = tqdm(
                    total=self._cv_splits_hp,
                    desc="Assessing the Cross-Validation",
                    leave=False,
                    colour="green",
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
                save_hp_performance(
                    base_dir,
                    kwargs_model,
                    kwargs_scaler,
                    cv_metric_values,
                    cv_hist_values.T,
                    metric_names,
                    hp_step,
                    s_features,
                    s_primary,
                    s_labels,
                )
                hp_step += 1
                hp_bar.update()
        hp_bar.close()
        summary_outcomes = (
            "The grid search based on CV were done with"
            "\n\t[Avg {}] = {}"
            "\n\t[Features] {}"
            "\n\t[Primary] {}"
            "\n\t[Labels] {}"
            "\n\t[Best model parameters] {}"
            "\n\t[Best scaler parameters] {}".format(
                metric_names[0],
                best_metric_value[0],
                s_features,
                s_primary,
                s_labels,
                self._best_kwargs_model,
                self._best_kwargs_scaler,
            )
        )
        self._file_log.info(summary_outcomes)
        print("\n" + summary_outcomes + "\n")
        return (metric_names[0], best_metric_value[0])


class EnvironmentOptimizer(BagingModelOptimizer):
    def __init__(
        self,
        symbol,
        db_path,
        model_fn,
        primary_model_fn,
        labels_fn,
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
        preload={"time": [5, 10, 15, 30, 60, "day"]}
    ):
        self._labels_fn = labels_fn
        self._primary_model_fn = primary_model_fn
        self._cv_splits_hp = cv_splits_hp
        self._run_dir = os.path.join(
            log_dir, author, "run-" + dt.now().strftime("%Y%m%d-%H%M%S")
        )
        self._refined_data = RefinedData(symbol, db_path, preload=preload)
        self._file_log = create_log_file("environment", log_dir=self._run_dir)
        self._best_kwargs_model = None
        self._best_kwargs_scaler = None
        self._best_kwargs_labels = None
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
            metrics,
        )
        # TODO: Ensure that all functions to get features in refined data have the format "get_<NAME>"
        self._feature_acrons = {
            "MA": "get_simple_MA",
            "DEV": "get_deviation",
            "RSI": "get_RSI",
            "FRACDIFF": "frac_diff",
        }

    # TODO: Implement case 2
    def __set_env(
        self, kwargs_features, kwargs_primary, kwargs_labels, approach=1
    ):
        steps = ["[Labels]"]
        for feature_name in kwargs_features.keys():
            steps.append("[" + feature_name + "]")
        bar = tqdm(
            total=len(steps),
            desc="Setting Environment [Primary]",
            leave=False,
            colour="green",
        )
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
        bar.close()
        return horizon, closed, data, labels.values

    def __parser_features(self, kwargs_features):
        summary = {}
        new_kwargs = {}
        for k, v in kwargs_features.items():
            name = k.split("__")[0]
            param = k.split("__")[1]
            f_name = self._feature_acrons[name]
            if f_name not in new_kwargs:
                summary[name] = ""
                new_kwargs[f_name] = {}
            summary[name] += param[0] + ":" + str(v) + ","
            new_kwargs[f_name][param] = v
        str_summary = ""
        for name, s in summary.items():
            str_summary += "[" + name + "]" + s[0:-1]
        return new_kwargs, str_summary

    def __parser_primary(self, kwargs_primary):
        str_summary = ""
        new_kwargs = {}
        strategy = kwargs_primary.pop("strategy")
        str_summary += "s" + ":" + strategy + ","
        new_kwargs["strategy"] = strategy
        new_kwargs["features"] = {}
        for k, v in kwargs_primary.items():
            new_kwargs["features"][k] = v
            str_summary += k[0] + ":" + str(v) + ","
        return new_kwargs, str_summary[0:-1]

    # TODO: Maybe a better solution be not to use the dictionary in Labels initialization
    def __parser_labels(self, kwargs_labels):
        str_summary = ""
        new_kwargs = {"operation_parameters": {}}
        for k, v in kwargs_labels.items():
            new_kwargs["operation_parameters"][k] = v
            str_summary += k[0] + ":" + str(v) + ","
        return new_kwargs, str_summary[0:-1]

    def train(
        self,
        kwargs_model=None,
        kwargs_scaler=None,
        kwargs_features=None,
        kwargs_primary=None,
        kwargs_labels=None,
    ):
        if kwargs_features is None and self._best_kwargs_features is None:
            raise ValueError(
                "At least one of the values, kwargs_features or"
                " _best_kwargs_features, have to be setted"
            )
        elif kwargs_features is None and self._best_kwargs_features is not None:
            kwargs_features = self._best_kwargs_features.copy()
        if kwargs_primary is None and self._best_kwargs_primary is None:
            raise ValueError(
                "At least one of the values, kwargs_primary or"
                " _best_kwargs_primary, have to be setted"
            )
        elif kwargs_primary is None and self._best_kwargs_primary is not None:
            kwargs_primary = self._best_kwargs_primary.copy()
        if kwargs_labels is None and self._best_kwargs_labels is None:
            raise ValueError(
                "At least one of the values, kwargs_labels or"
                " _best_kwargs_labels, have to be setted"
            )
        elif kwargs_labels is None and self._best_kwargs_labels is not None:
            kwargs_labels = self._best_kwargs_labels.copy()

        horizon, closed, data, labels = self.__set_env(
            kwargs_features, kwargs_primary, kwargs_labels
        )
        model, scaler, cv_predictions, metrics = self.cv_fit(
            data, labels, closed, horizon
        )
        return model, scaler, cv_predictions

    def env_optimize(
        self, hp_model, hp_features, hp_primary, hp_labels, hp_scaler=None
    ):
        if hp_scaler is None:
            hp_scaler = {}
        grid_features = ParameterGrid(hp_features)
        grid_primary = ParameterGrid(hp_primary)
        grid_labels = ParameterGrid(hp_labels)

        n_steps = 1
        best_metric = 0
        for d in hp_features:
            for v in d.values():
                n_steps *= len(v)
        for d in hp_primary:
            for v in d.values():
                n_steps *= len(v)
        for v in hp_labels.values():
            n_steps *= len(v)
        hp_bar = tqdm(
            total=n_steps, desc="Mapping the Features", colour="green"
        )
        for kwargs_features in grid_features:
            kwargs_features, s_features = self.__parser_features(
                kwargs_features
            )
            for kwargs_primary in grid_primary:
                kwargs_primary, s_primary = self.__parser_primary(
                    kwargs_primary
                )
                for kwargs_labels in grid_labels:
                    kwargs_labels, s_labels = self.__parser_labels(
                        kwargs_labels
                    )
                    horizon, closed, data, labels = self.__set_env(
                        kwargs_features, kwargs_primary, kwargs_labels
                    )
                    out_metric = self.hp_meta_optimize(
                        data,
                        labels,
                        horizon,
                        closed,
                        hp_model,
                        hp_scaler,
                        s_features,
                        s_primary,
                        s_labels,
                    )
                    if out_metric[1] > best_metric:
                        best_metric = out_metric[1]
                        best_kwargs_labels = kwargs_labels
                        best_kwargs_primary = kwargs_primary
                        best_kwargs_features = kwargs_features
                        best_kwargs_model = self._best_kwargs_model
                        best_kwargs_scaler = self._best_kwargs_scaler
                    self._best_kwargs_model = None
                    self._best_kwargs_scaler = None
                    hp_bar.update()
        hp_bar.close()
        self._best_kwargs_model = best_kwargs_model
        self._best_kwargs_scaler = best_kwargs_scaler
        self._best_kwargs_labels = best_kwargs_labels
        self._best_kwargs_primary = best_kwargs_primary
        self._best_kwargs_features = best_kwargs_features
        return self.train()
