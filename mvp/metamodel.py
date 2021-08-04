import os
import json
import inspect
import shutil
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

from sklearn.metrics import f1_score, balanced_accuracy_score, precision_score

from mvp.labels import event_label_series
from mvp.draw import draw_roc_curve
from mvp.selection import (
    count_occurrences,
    avg_uniqueness,
    time_weights,
    sample_weights,
    raw_horizon,
    PEKFold,
)


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
    """Save the hyper-parameters using Tensorboard"""
    with summary.create_file_writer(base_dir).as_default():
        hparams = {}
        hparams.update(kwargs_model)
        hparams.update(kwargs_scaler)
        if s_features is not None:
            # hparams["features"] = s_features
            hparams.update(s_features)
        # if s_primary is not None:
        #     hparams.update(s_primary)
        if s_labels is not None:
            # hparams["labels"] = s_labels
            hparams.update(s_labels)
        hp.hparams(hparams)
        for name, value in avg_metrics.items():
            summary.scalar(name, value, step=step)
        for name, value in zip(metric_names, hist_data):
            summary.histogram(name, value, step=step)


def create_log_file(name, log_dir):
    """Create a file in `log_dir` with name `name` to save the logs"""
    os.makedirs(log_dir, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, "{}.log".format(name)),
        maxBytes=200 * 1024 * 1024,
        backupCount=1,
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


class BaggingModelOptimizer:
    """
    Optimize any Bagging model based on sklearn classes.

    Parameters
    ----------
    `model_fn` : ``callable``
        A callable object that instantiates a baging model.
    `run_dir` : ``str``
        The path in which the outcomes from optimization will be saved.
    `outputs` : ``str``
        A string to describe the expected output model. It is either `category` or
        `probability`.
    `threshold` : ``float``
        The value of the threshold used to classify an event based on the probability
        as a positive label [ignored in `category` mode].
    `scaler_pipeline` : ``Pipeline``
        A sklearn ``Pipeline`` composed of the steps to preprocess data.
    `cv_splits_fit` : ``int``
        The number of folds that will be used in the CV during the final fitting.
    `cv_splits_hp` : ``int``
        The number of folds that will be used in the CV.
    `use_weight` : ``bool``
        Whether the sample weights are used or not.
    `num_of_threads` : ``int``
        Number of threads to be used in the training.
    `min_time_weight` : ``float``
        The minimum value assigned to the weight based on timestamp
        (only used if `use_weight=True`)
    `time_weights_fn` : ``callable``
        A callable to create the weights based on timestamp.
    `samples_weights_fn` : ``callable``
        A callable to create the weights based on horizon overlapping and return values.
    `metrics` : ``dict``
        The metrics to be reported during the optimization process. The first metric
        is used to guide the selection of the best set of hyper-parameters.
        Each key of the dictionary must be the name of the metric, and the value
        must be a ``tuple`` with two elements, a ``callabel`` used to apply the metric
        and a ``bool`` to indicate whether the sample weights will be used or not
        to assess the metric.
    `verbose` : ``int``
        It is either 0 or 1. It 1, some log information will be shown in the terminal.
        Regardless the verbose value, these information is always accessed in log files.
    `seed` : ``int``
        The seed to initiate the numpy ``RandomState``.

    """

    def __init__(
        self,
        model_fn,
        run_dir,
        outputs="probability",
        threshold=0.6,
        scaler_pipeline=None,
        cv_splits_fit=10,
        cv_splits_hp=5,
        use_weight=True,
        num_of_threads=None,
        min_time_weight=0.25,
        time_weights_fn=time_weights,
        samples_weights_fn=sample_weights,
        metrics={
            "Precision": precision_score,
            "F1": f1_score,
            "BAcc": balanced_accuracy_score,
        },
        verbose=0,
        seed=12345,
    ):
        self._hp_steps = 0
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
        self._meta_file_log = create_log_file(
            "metamodel", log_dir=self._run_dir
        )
        if self._use_weight:
            if (self._time_weights_fn is None) and (
                self._samples_weights_fn is None
            ):
                raise ValueError(
                    "With `use_weight` is True, at least one of the functions,"
                    " `time_weight_fn` or `sample_weights_fn`, have to be defined"
                )
        if not (
            hasattr(self._model_fn, "predict")
            and hasattr(self._model_fn, "predict_proba")
        ):
            raise ValueError(
                "The model {} doesn't have one or both of the"
                " methods predict and predict_proba".format(self._model_fn)
            )
        if outputs == "probability":
            self._threshold = threshold
            self._predict_fn_name = "predict_proba"
        else:
            self._predict_fn_name = "predict"
        self._verbose = verbose
        self._seed = seed
        self._random_state = np.random.RandomState(seed)

    def __is_better(self, last, best):
        """Determine if the set of values in `last` is greater than `best`"""
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
    ):
        """Initiate the model and the scaler based on the kwargs, the weights are also initiated"""
        if kwargs_scaler is None:
            kwargs_scaler = {}
        if self._scaler_pipeline is not None:
            scaler = self._scaler_pipeline.set_params(**kwargs_scaler)
        else:
            scaler = None
        if self._use_weight:
            train_weight = self.get_weight(closed, train_horizon)
        else:
            train_weight = None
        model = self._model_fn(
            n_jobs=self._num_of_threads,
            n_bootstrap_jobs=self._num_of_threads,
            random_state=self._random_state,
            **kwargs_model,
        )
        return model, scaler, train_weight

    def __partial_fit_eval(
        self,
        data,
        labels,
        train_idx,
        test_idx,
        horizon,
        np_horizon,
        closed,
        test_weight,
        kwargs_model,
        kwargs_scaler=None,
    ):
        """Fit and evaluate the model based on partial configuration"""
        model, scaler, train_weight = self.__set_model(
            kwargs_scaler,
            kwargs_model,
            closed,
            horizon[train_idx],
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
        predicted_proba = model.predict_proba(x_test)

        predicted = model.__getattribute__(self._predict_fn_name)(x_test)
        if hasattr(self, "_threshold"):
            predicted = (predicted[:, 1] >= self._threshold).astype(int)
        for name, metric_fn in self._metrics.items():
            outcomes[name] = metric_fn(
                y_test, predicted, sample_weight=test_weight
            )
        return outcomes, predicted, horizon.index[test_idx], predicted_proba

    def __save_best_info(self, model, cv_predictions, base_dir, **kwargs):
        dump(model, os.path.join(base_dir, "fitted_model.pkl"))
        dump(cv_predictions, os.path.join(base_dir, "metamodel_triggers.pkl"))
        with open(os.path.join(base_dir, "best_run_params.json"), "w") as f:
            json.dump(kwargs, f)
            print(os.path.join(base_dir, "best_run_params.json"))

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
        """Fit the model using all training data"""
        print("Training the last model with all training dataset...  ", end="")
        model, scaler, train_weight = self.__set_model(
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
        print("DONE")
        return model, scaler

    def get_weight(self, closed, horizon, min_chunk_size=100):
        """Get the sample weights based on timestamp, horizon overlappig, and return values"""
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
        weights,
        kwargs_model=None,
        kwargs_scaler=None,
        s_features=None,
        s_primary=None,
        s_labels=None,
    ):
        """Fit the model using a pre-defined configuration and report statistics using CV"""
        if kwargs_model is None and self._best_kwargs_model is None:
            raise ValueError(
                "At least one of the values, kwargs_model or"
                " _best_kwargs_model, have to be set"
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
            position=0,
            desc="Assessing CV for the Best Configuration",
            colour="green",
        )

        cv_step = 0
        cv_predictions = []
        cv_expected = []
        cv_true_probs = []
        spliter = pekfold.split(horizon)
        for train_idx, test_idx in spliter:
            outcomes, predicted, index, probs = self.__partial_fit_eval(
                data,
                labels,
                train_idx,
                test_idx,
                horizon,
                np_horizon,
                closed,
                weights[test_idx],
                kwargs_model,
                kwargs_scaler,
            )
            partial_metric_value = np.array(list(outcomes.values()))
            cv_hist_values[cv_step] = partial_metric_value
            cv_metric_values += partial_metric_value
            cv_step += 1
            cv_bar.update()
            cv_bar.set_postfix(**outcomes)
            cv_predictions.append(pd.Series(predicted, index=index))
            cv_expected.append(labels[test_idx])
            cv_true_probs.append(probs[:, 1])
        cv_bar.close()
        cv_metric_values /= self._cv_splits_fit
        cv_metric_values = dict(
            zip(
                ["avg_" + name for name in metric_names],
                cv_metric_values,
            )
        )
        save_hp_performance(
            os.path.join(base_dir, "best_hp_set"),
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
        draw_roc_curve(base_dir, cv_expected, cv_true_probs)
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
        self._meta_file_log.info(summary_outcomes)
        model, scaler = self.__fit(
            data,
            labels,
            horizon,
            np_horizon,
            closed,
            kwargs_model,
            kwargs_scaler,
        )
        self.__save_best_info(
            model,
            cv_predictions,
            base_dir,
            metrics=cv_metric_values,
            feature_params=s_features,
            primary_params=s_primary,
            label_params=s_labels,
            model_params=kwargs_model,
            scaler_params=kwargs_scaler,
        )
        if scaler is not None:
            dump(scaler, os.path.join(base_dir, "fitted_scaler.pkl"))
        return model, scaler, cv_predictions, cv_metric_values

    def hp_meta_optimize(
        self,
        data,
        labels,
        horizon,
        closed,
        weights,
        hp_model,
        hp_scaler=None,
        s_features=None,
        s_primary=None,
        s_labels=None,
    ):
        """Apply the grid search to determine the best hyper parameter for the model"""
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
            position=1,
            leave=False,
            desc="Searching into Hyper-Parameter Grid",
            colour="red",
        )

        metric_names = list(self._metrics.keys())
        best_metric_value = np.zeros(len(self._metrics))
        pekfold = PEKFold(n_splits=self._cv_splits_hp)
        for kwargs_model in grid_model:
            for kwargs_scaler in grid_scaler:
                cv_step = 0
                cv_bar = tqdm(
                    total=self._cv_splits_hp,
                    desc="Assessing the Cross-Validation",
                    position=2,
                    leave=False,
                    colour="blue",
                )
                cv_hist_values = np.zeros(
                    (self._cv_splits_hp, len(metric_names))
                )
                cv_metric_values = np.zeros(len(self._metrics))
                spliter = pekfold.split(horizon)
                for train_idx, test_idx in spliter:
                    outcomes, _, _, _ = self.__partial_fit_eval(
                        data,
                        labels,
                        train_idx,
                        test_idx,
                        horizon,
                        np_horizon,
                        closed,
                        weights[test_idx],
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
                    os.path.join(base_dir, "hp_set-" + str(self._hp_steps)),
                    kwargs_model,
                    kwargs_scaler,
                    cv_metric_values,
                    cv_hist_values.T,
                    metric_names,
                    self._hp_steps,
                    s_features,
                    s_primary,
                    s_labels,
                )
                self._hp_steps += 1
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
        self._meta_file_log.info(summary_outcomes)
        if self._verbose > 0:
            print("\n" + summary_outcomes + "\n")
        return ("Avg_" + metric_names[0], best_metric_value[0])


class EnvironmentOptimizer(BaggingModelOptimizer):
    """
    Optimize the finance environment selecting the sub-optimal statistical features and
    hyper-parameters for a defined Bagging model based on sklearn classes.

    Parameters
    ----------
    `refined_data` : ``RefinedData``
        A instance of ``RefinedData``.
    `model_fn` : ``callable``
        A callable object that instantiates a baging model.
    `primary_model_fn` : ``callable``
        A callable object that instantiates the primary model.
    `author` : ``str``
        The author name.
    `outputs` : ``str``
        A string to describe the expected output model. It is either `category` or
        `probability`.
    `threshold` : ``float``
        The value of the threshold used to classify any event based on the probability
        as a positive label [ignored in `category` mode].
    `scaler_pipeline` : ``Pipeline``
        A sklearn ``Pipeline`` composed of the steps to preprocess data.
    `cv_splits_fit` : ``int``
        The number of folds that will be used in the CV during the final fitting.
    `cv_splits_hp` : ``int``
        The number of folds that will be used in the CV.
    `use_weight` : ``bool``
        Whether the sample weights are used or not.
    `num_of_threads` : ``int``
        Number of threads to be used in the training.
    `min_time_weight` : ``float``
        The minimum value assigned to the weight based on timestamp
        (only used if `use_weight=True`)
    `time_weights_fn` : ``callable``
        A callable to create the weights based on timestamp.
    `samples_weights_fn` : ``callable``
        A callable to create the weights based on horizon overlapping and return values.
    `log_dir` : ``str``
        The path to the main directory to save all assets.
    `metrics` : ``dict``
        The metrics to be reported during the optimization process. The first metric
        is used to guide the selection of the best set of hyper-parameters.
        Each key of the dictionary must be the name of the metric, and the value
        must be a ``tuple`` with two elements, a ``callabel`` used to apply the metric
        and a ``bool`` to indicate whether the sample weights will be used or not
        to assess the metric.
    `approach` : ``int``
        It is either 0 or 1. If 0, the side will be considered in the feature space and
        labels will be interpreted as with profit and without for 1 and 0. On the other
        hand, the approach 1 doesn't consider the sides and label 1 means "reached TP
        bar first into horizon" and 0 means "reached SL first into horizon".
    `verbose` : ``int``
        It is either 0 or 1. If 1, some log information will be shown in the terminal.
        Regardless the verbose value, these information is always accessed in log files.
    `seed` : ``int``
        The seed to initiate the numpy ``RandomState``.

    """

    def __init__(
        self,
        refined_data,
        model_fn,
        primary_model_fn,
        author,
        outputs="probability",
        threshold=0.6,
        cv_splits_fit=10,
        cv_splits_hp=5,
        use_weight=True,
        num_of_threads=None,
        scaler_pipeline=None,
        min_time_weight=0.25,
        time_weights_fn=time_weights,
        samples_weights_fn=sample_weights,
        log_dir="logs",
        metrics={
            "Precision": precision_score,
            "F1": f1_score,
            "BAcc": balanced_accuracy_score,
        },
        approach=0,
        verbose=0,
        seed=12345,
    ):
        run_dir = os.path.join(
            log_dir, author, "run-" + dt.now().strftime("%Y%m%d-%H%M%S")
        )
        os.makedirs(run_dir, exist_ok=True)
        frame = inspect.stack()[1]
        module = inspect.getmodule(frame[0])
        filename = module.__file__
        shutil.copyfile(
            os.path.abspath(filename), os.path.join(run_dir, filename)
        )
        self._author = author
        self._primary_model_fn = primary_model_fn
        self._refined_data = refined_data
        self._approach = approach
        self._env_file_log = create_log_file("environment", log_dir=run_dir)
        self._best_kwargs_labels = None
        self._best_kwargs_primary = None
        self._best_kwargs_features = None
        super(EnvironmentOptimizer, self).__init__(
            model_fn,
            run_dir,
            outputs,
            threshold,
            scaler_pipeline,
            cv_splits_fit,
            cv_splits_hp,
            use_weight,
            num_of_threads,
            min_time_weight,
            time_weights_fn,
            samples_weights_fn,
            metrics,
            verbose,
            seed,
        )

    def __set_env(self, kwargs_features, kwargs_primary, kwargs_labels):
        """Set the finance environment"""
        process_steps = ["[Labels]", "[Weights for Test]"]
        for feature_name in kwargs_features.keys():
            process_steps.append("[" + feature_name + "]")
        bar = tqdm(
            total=len(process_steps),
            desc="Setting Environment [Primary]",
            leave=False,
            colour="red",
        )
        process_steps = iter(process_steps)

        step = 1
        if "kwargs" in kwargs_primary and "step" in kwargs_primary["kwargs"]:
            step = kwargs_primary["kwargs"]["step"]
        raw_data = self._refined_data.time_bars(step=step)
        closed = raw_data.loc[:, "Close"]

        bar.update()
        bar.set_description(
            "Setting Environment {}".format(next(process_steps))
        )

        complete_events = event_label_series(
            self._primary_model_fn(self._refined_data, **kwargs_primary),
            raw_data,
            **kwargs_labels,
        )

        zero_mask = complete_events["Label"] != 0
        complete_events = complete_events.loc[zero_mask]

        sides = complete_events["Side"]
        labels = complete_events["Label"]
        horizon = pd.Series(
            complete_events["PositionEnd"], index=complete_events.index
        )

        bar.update()
        bar.set_description(
            "Setting Environment {}".format(next(process_steps))
        )

        weights = self.get_weight(closed, horizon)

        stats = []
        latest_start = closed.index[0]
        for get_stat_name, kwargs in kwargs_features.items():
            bar.update()
            bar.set_description(
                "Setting Environment {}".format(next(process_steps))
            )
            feature = self._refined_data.__getattribute__(get_stat_name)(
                **kwargs
            )
            stats.append(feature)
            if feature.index[0] > latest_start:
                latest_start = feature.index[0]

        event_index = horizon.index
        update_horizon = latest_start > event_index[0]
        if update_horizon:
            mask = event_index >= latest_start
            horizon = horizon.loc[mask]
            sides = sides.loc[mask]
            labels = labels.loc[mask]
            event_index = horizon.index
            self._env_file_log.warning(
                "{} events were droped due to {} with {}"
                " starts before the first event".format(
                    (~mask).sum(), get_stat_name, kwargs
                )
            )
            for i, feature in enumerate(stats):
                stats[i] = feature.loc[latest_start:]
        for i, feature in enumerate(stats):
            stats[i] = feature.loc[event_index].values

        if self._approach == 0:
            stats.append(sides.values)
        else:
            labels = pd.Series(
                labels.values * sides.values, index=labels.index
            )
        labels.replace(-1, 0, inplace=True)
        data = np.stack(stats, axis=1)

        self._env_file_log.info(
            "{} of {} events with label ZERO were droped".format(
                (~zero_mask).sum(), zero_mask.shape[0]
            )
        )
        self._env_file_log.info(
            "The environment was set with {} prices and {} events"
            " such that {} are labeled as 1 and {} as -1".format(
                closed.shape[0],
                horizon.shape[0],
                (labels == 1).sum(),
                (labels == 0).sum(),
            )
        )
        bar.close()
        return horizon, closed, data, labels.values, weights

    def __parser_features(self, kwargs_features):
        new_kwargs = {}
        for k, v in kwargs_features.items():
            name = k.split("__")[0]
            param = k.split("__")[1]
            f_name = "get_" + name
            if f_name not in new_kwargs:
                new_kwargs[f_name] = {}
            new_kwargs[f_name][param] = v
        return new_kwargs

    def train(
        self,
        kwargs_model=None,
        kwargs_features=None,
        kwargs_primary=None,
        kwargs_labels=None,
        kwargs_scaler=None,
        s_features=None,
        s_primary=None,
        s_labels=None,
    ):
        """Train the model for a pre-defined configuration"""
        if kwargs_features is None and self._best_kwargs_features is None:
            raise ValueError(
                "At least one of the values, kwargs_features or"
                " _best_kwargs_features, have to be set"
            )
        elif (
            kwargs_features is None and self._best_kwargs_features is not None
        ):
            kwargs_features = self._best_kwargs_features.copy()
            s_features = self._best_s_features
        if kwargs_primary is None and self._best_kwargs_primary is None:
            raise ValueError(
                "At least one of the values, kwargs_primary or"
                " _best_kwargs_primary, have to be set"
            )
        elif kwargs_primary is None and self._best_kwargs_primary is not None:
            kwargs_primary = self._best_kwargs_primary.copy()
            s_primary = self._best_s_primary
        if kwargs_labels is None and self._best_kwargs_labels is None:
            raise ValueError(
                "At least one of the values, kwargs_labels or"
                " _best_kwargs_labels, have to be set"
            )
        elif kwargs_labels is None and self._best_kwargs_labels is not None:
            kwargs_labels = self._best_kwargs_labels.copy()
            s_labels = self._best_s_labels

        horizon, closed, data, labels, weights = self.__set_env(
            kwargs_features,
            kwargs_primary,
            kwargs_labels,
        )
        model, scaler, cv_predictions, metrics = self.cv_fit(
            data,
            labels,
            closed,
            horizon,
            weights,
            kwargs_model,
            kwargs_scaler,
            s_features,
            s_primary,
            s_labels,
        )
        return model, scaler, cv_predictions

    def env_optimize(
        self, hp_model, hp_features, hp_primary, hp_labels, hp_scaler=None
    ):
        """
        Apply the grid search to determine the optimal parameters for features,
        the primary and the labels generator.

        If the use of same features with different parameters is desired, add a suffix
        to the desired feature key, e.g.,
            ```
            dict(sma__window__A=[100], sma__window__A=[10000])
            ```
        """
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
        for v in hp_primary:
            n_steps *= len(v)
        for v in hp_labels.values():
            n_steps *= len(v)
        hp_bar = tqdm(
            total=n_steps,
            position=0,
            leave=False,
            desc="Mapping the Features",
            colour="green",
        )
        for s_features in grid_features:
            kwargs_features = self.__parser_features(s_features)
            for kwargs_primary in grid_primary:
                for kwargs_labels in grid_labels:
                    horizon, closed, data, labels, weights = self.__set_env(
                        kwargs_features, kwargs_primary, kwargs_labels
                    )
                    out_metric = self.hp_meta_optimize(
                        data,
                        labels,
                        horizon,
                        closed,
                        weights,
                        hp_model,
                        hp_scaler,
                        s_features,
                        kwargs_primary,
                        kwargs_labels,
                    )
                    if out_metric[1] > best_metric:
                        best_metric = out_metric[1]
                        best_kwargs_labels = kwargs_labels
                        best_kwargs_primary = kwargs_primary
                        best_kwargs_features = kwargs_features
                        best_kwargs_model = self._best_kwargs_model
                        best_kwargs_scaler = self._best_kwargs_scaler
                        best_s_labels = kwargs_labels
                        best_s_primary = kwargs_primary
                        best_s_features = s_features
                    self._best_kwargs_model = None
                    self._best_kwargs_scaler = None
                    self._hp_steps += 1
                    hp_bar.set_postfix(**{out_metric[0]: out_metric[1]})
                    hp_bar.update()
        hp_bar.close()
        self._best_kwargs_model = best_kwargs_model
        self._best_kwargs_scaler = best_kwargs_scaler
        self._best_kwargs_labels = best_kwargs_labels
        self._best_kwargs_primary = best_kwargs_primary
        self._best_kwargs_features = best_kwargs_features
        self._best_s_labels = best_s_labels
        self._best_s_primary = best_s_primary
        self._best_s_features = best_s_features
        model, scaler, cv_predictions = self.train()
        if self._verbose > 0:
            print(
                "\n The K-fold cross-validation training"
                " with grid search was finished."
                "\n All logs are saved in files {} and {},"
                " you also can use Tensorboard to "
                "analyse the outcomes, to do so, make"
                "\n `tensorboard --logdir {}`".format(
                    os.path.join(self._run_dir, "environment.log"),
                    os.path.join(self._run_dir, "metamodel.log"),
                    self._run_dir,
                )
            )
        return model, scaler, cv_predictions
