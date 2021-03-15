import pandas as pd
import mvp
import numpy as np
import datetime as dt
from mvp.refined_data import RefinedData


class PrimaryModel:
    """
        This class implements event-driven trading strategies, which generate Buy/Sell triggers whenever certain conditions are met
    by the parameters/indicators. It shoud be understood as being defined by a model-type (so far only 3 supported), and the parameters
    for the given model type. The class then generates the trading signals automatically by evoking the `PrimaryModel.events()` method and
    storing the information in the `.PrimaryModel.events_df` attribute.
        This class uses refined_data data to process three models:

    - Crossing Averages Model
    - Bollinger Bands Model
    - Classical Filter Model

    Parameters
    ----------
    `refined_data` : ``refined_data.RefinedData Object``
        A RefinedData object from the file refined_data.py
    `strategy` : ``str``
        Three available types: ``crossing-MA``, `bollinger-bands``, ``classical-filter``
    `parameters` : ``dict```
        A dict containing model parameters keys:
            For Crossing Averages Model it should be like:
                {'MA':[500,1000]}
            For Bollinger Bands Model it should be like:
                {'MA':500,'DEV':20,'K_value':2}
            For Classical Model it should be like
                {'threshold': 0.01}

    Usage
    -----
    The user should provide a RefinedData object contaning no a priori calculated statistics (only 'OHLC', Volume, TickVol) along with the
    desired model-type and its parameters.  This class is intended to be used in the following flow:

    Given a .db containing ('OHLC', Volume, TickVol) time series data, one should instantiate a RefinedData to read the time-series and feed
    the PrimaryModel class. The class then calculates the necessary features for the desired strategy in a curated data in the `PrimaryModel.feature_data`
    attribute. The feature_data is then used to generate the event triggers, labeled by the appropriate Side of
    the market, stored in PrimaryModel.events.
    """

    def __init__(self, refined_data, strategy, features):
        self.__features = {
            "MA": "get_simple_MA",
            "DEV": "get_deviation",
            "RSI": "get_RSI",
            "FRAC_DIFF": "frac_diff",
            "AUTOCORRELATION": "moving_autocorr",
            "AUTOCORRELATION_PERIOD": "autocorr_period",
        }
        self.strategy = strategy
        self.features = features
        self.feature_data = self.get_feature_data(refined_data)
        self.events = self.get_events_dataframe()[["Side"]].dropna()

    def get_feature_data(self, refined_data):
        dataframe_list = [refined_data.df[["Close"]]]
        for feature in self.features.keys():
            if feature not in self.__features.keys():
                continue
            if isinstance(self.features[feature], list):
                parameters_list = self.features[feature]
                for parameter in parameters_list:
                    try:
                        if isinstance(parameter, tuple):
                            dataframe_list.append(
                                pd.DataFrame(
                                    refined_data.__getattribute__(
                                        self.__features[feature]
                                    )(*parameter).rename(
                                        feature + "_" + str(parameter)
                                    )
                                )
                            )
                        else:
                            dataframe_list.append(
                                pd.DataFrame(
                                    refined_data.__getattribute__(
                                        self.__features[feature]
                                    )(parameter).rename(
                                        feature + "_" + str(parameter)
                                    )
                                )
                            )
                    except Exception as err:
                        print(err, ": param {} given".format(parameter))
            else:
                parameter = self.features[feature]
                try:
                    if isinstance(parameter, tuple):
                        dataframe_list.append(
                            pd.DataFrame(
                                refined_data.__getattribute__(
                                    self.__features[feature]
                                )(*parameter).rename(
                                    feature + "_" + str(parameter)
                                )
                            )
                        )
                    else:
                        dataframe_list.append(
                            pd.DataFrame(
                                refined_data.__getattribute__(
                                    self.__features[feature]
                                )(parameter).rename(
                                    feature + "_" + str(parameter)
                                )
                            )
                        )
                except ValueError as err:
                    print(err, ": param {} given".format(parameter))
        feature_df = pd.concat(dataframe_list, axis=1)
        return feature_df

    def sign(self, x):
        if x >= 0:
            return 1
        else:
            return -1

    def states_condition(self, x):
        x = x.tolist()
        close = x[0]
        plusK = x[1]
        minusK = x[2]

        if close < plusK and close > minusK:
            return 0
        if close >= plusK:
            return 1
        if close <= minusK:
            return -1

        return np.nan

    def events_crossing_MA(self):
        """
        Public method which computes the events triggered by a crossing moving average trading strategy.
        Given that the PrimaryModel class was initialized with two MA parameters, this method identifies
        the pd.Timestamp() value of a crossing of averages as well as the side signal: Buy or Sell,
        depending on the direction of the crossing. A pd.Dataframe() indexed by pd.Timestamp() and valued
        in a Boolean field (Buy: 1, Sell: -1) is returned.

        Parameters
        ----------
        None

        Modified
        ----------
        None

        Return
        ----------

        ``pd.DataFrame()``
            columns = ['DateTime','Side']
            'DateTime' is the pd.index column, containing pd.Timestamp() values for the calculated events.
            'Side' is a coulmn containing the side of the stratgey: Buy = 1, Sell = -1.
        """
        MA_params = list(set(self.features["MA"]))
        if len(MA_params) > 2:
            print(
                "warning, Number of parameters exceeded maximum of two, using MA's: ["
                + str(MA_params[0])
                + ", "
                + str(MA_params[1])
                + "]"
            )
        delta_MA = pd.DataFrame()
        delta_MA["Delta"] = (
            self.feature_data["MA_" + str(MA_params[0])]
            - self.feature_data["MA_" + str(MA_params[1])]
        ).dropna()
        events_MA = delta_MA["Delta"].apply(self.sign)
        events_MA = (
            events_MA.rolling(window=2)
            .apply(
                lambda x: -1
                if x[0] > x[1]
                else (1 if x[0] < x[1] else np.nan),
                raw=True,
            )
            .dropna()
        )
        return (
            pd.DataFrame(events_MA)
            .rename(columns={"Delta": "Side"})
            .reset_index()
        )

    def events_classical_filter(self, threshold=0.1):
        """
        Public method which computes the events triggered by a Classical Filter trading strategy. This method firs identifies all
        local inflexions of 'Close' prices of a given asset. It then calculates the CUSUM of returns from an inflexion and generates an
        event trigger for the first pd.Timestamp() value for which the CUSUM > threshold. The side of the event is a Buy if the inflexion
        is a minimum, and a Sell if it is a maximum. A pd.Dataframe() indexed by pd.Timestamp() and valued in a Bool (Buy: 1, Sell: -1)
        is returned.

        Parameters
        ----------
        ``threshold``: 'float'
            defines the threshold for the CUSUM operation. When CUSUM achieves threshold, an event is triggered.

        Modified
        ----------
        None

        Return
        ----------

        ``pd.DataFrame()``
            columns = ['DateTime','Side']
            'DateTime' is the pd.index column, containing pd.Timestamp() values for the calculated events.
            'Side' is a coulmn containing the side of the stratgey: Buy = 1, Sell = -1.
        """
        close_data = self.feature_data[["Close"]]
        close_data["Min"] = close_data[
            (close_data["Close"].shift(1) > close_data["Close"])
            & (close_data["Close"].shift(-1) > close_data["Close"])
        ]["Close"]
        close_data["Max"] = close_data[
            (close_data["Close"].shift(1) < close_data["Close"])
            & (close_data["Close"].shift(-1) < close_data["Close"])
        ]["Close"]
        saddle_events = (
            (close_data[close_data["Max"] >= 0].index)
            .append(close_data[close_data["Min"] >= 0].index)
            .sort_values()
        )
        if close_data.loc[saddle_events[0]]["Max"] != np.nan:
            side = -1
        else:
            side = 1
        events_CF = []
        for start, end in list(zip(saddle_events, saddle_events[1:])):
            inter_saddle_df = close_data[start:end]
            inter_saddle_df["Returns"] = (
                inter_saddle_df["Close"] / inter_saddle_df["Close"].shift(1)
                - 1
            )
            inter_saddle_df["Cusum"] = 0
            if side == 1:
                for i in range(1, len(inter_saddle_df["Cusum"].values)):
                    inter_saddle_df["Cusum"].iloc[i] = max(
                        (
                            inter_saddle_df["Cusum"].iloc[i - 1]
                            + inter_saddle_df["Returns"].iloc[i]
                        ),
                        0,
                    )
                occurances = inter_saddle_df[
                    inter_saddle_df["Cusum"] > threshold
                ]
            if side == -1:
                for i in range(1, len(inter_saddle_df["Cusum"].values)):
                    inter_saddle_df["Cusum"].iloc[i] = -max(
                        -(
                            inter_saddle_df["Cusum"].iloc[i - 1]
                            + inter_saddle_df["Returns"].iloc[i]
                        ),
                        0,
                    )
                occurances = inter_saddle_df[
                    inter_saddle_df["Cusum"] < -threshold
                ]
            if not occurances.empty:
                event = [occurances.index[0], side]
                events_CF.append(event)
            side = -side

        return pd.DataFrame(events_CF, columns=["DateTime", "Side"])

    def events_bollinger(self):
        """
        Public method which computes the events triggered by a Bollinger Bands trading strategy. Given that the PrimaryModel
        class was initialized with a DEV, MA, and K_value parameter, this method identifies the pd.Timestamp() value of a crossing
        of the close price with respect to the upper and lower Bollinger Band, which gives us the side signal: Buy or Sell,
        depending on the crossed band. A pd.Dataframe() indexed by pd.Timestamp() and valued in a Bool (Buy: 1, Sell: -1)
        is returned.

        Parameters
        ----------
        None

        Modified
        ----------
        None

        Return
        ----------

        ``pd.DataFrame()``
            columns = ['DateTime','Side']
            'DateTime' is the pd.index column, containing pd.Timestamp() values for the calculated events.
            'Side' is a coulmn containing the side of the stratgey: Buy = 1, Sell = -1.
        """
        if isinstance(self.features["MA"], list):
            if len(self.features["MA"]) > 1:
                print(
                    "Warning, more than one MA provided, using MA "
                    + str(self.features["MA"][0])
                )
            MA_param = self.features["MA"][0]
        else:
            MA_param = self.features["MA"]

        if isinstance(self.features["DEV"], list):
            if len(self.features["DEV"]) > 1:
                print(
                    "Warning, more than one DEV provided, using DEV "
                    + str(self.features["DEV"][0])
                )
            DEV_param = self.features["DEV"][0]
        else:
            DEV_param = self.features["DEV"]
        K_value = self.features["K_value"]
        temp = pd.DataFrame()
        temp["Close"] = self.feature_data["Close"]
        temp["plusK"] = (
            self.feature_data["MA_" + str(MA_param)]
            + K_value * self.feature_data["DEV_" + str(DEV_param)]
        ).dropna()
        temp["minusK"] = (
            self.feature_data["MA_" + str(MA_param)]
            - K_value * self.feature_data["DEV_" + str(DEV_param)]
        ).dropna()

        temp["Side"] = temp.apply(self.states_condition, raw=True, axis=1)

        events_bol = (
            temp["Side"]
            .rolling(window=2)
            .apply(
                lambda x: -1
                if x[0] == 0 and x[1] == 1
                else (1 if x[0] == 0 and x[1] == -1 else np.nan),
                raw=True,
            )
        ).dropna()
        return pd.DataFrame(events_bol).reset_index()

    def events(self):
        if self.strategy == "crossing-MA":
            return self.events_crossing_MA()
        elif self.strategy == "bollinger-bands":
            return self.events_bollinger()
        elif self.strategy == "classical-filter":
            return self.events_classical_filter(self.features["threshold"])

    def get_events_dataframe(self):
        return pd.concat(
            [
                self.feature_data[["Close"]],
                self.events().set_index("DateTime"),
            ],
            axis=1,
        )