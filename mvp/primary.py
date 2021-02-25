import pandas as pd
import mvp
import numpy as np
import datetime as dt


class PrimaryModel:
    """
        This class implements event-driven trading strategies, which generate Buy/Sell triggers whenever certain conditions are met
    by the parameters/indicators. It shoud be understood as being defined by a model-type (so far only 3 supported), and the parameters
    for the given model type. The class then generates the trading signals automatically by evoking the `PrimaryModel.events()` method and
    storing the information in the `.PrimaryModel.events_df` attribute.
        This class uses curated data to process three models:

    - Crossing Averages Model
    - Bollinger Bands Model
    - Classical Filter Model

    Parameters
    ----------
    `raw_data` : ``RawData Object``
        A RawData object from the file rawdata.py
    `model_type` : ``str``
        Three available types: ``crossing-MA``, `bollinger-bands``, ``classical-filter``
    `parameters` : ``dict```
        A dict containing two keys: `ModelParameters` and `OperationParameters`
        Inside `ModelParameters` key we have to provide another dict in one of three available options:
            For Crossing Averages Model it should be like:
                {'MA':[500,1000]}
            For Bollinger Bands Model it should be like:
                {'MA':[500],'DEV':[20],'K_value':2}
            For Classical Model it should be like
                {'threshold': 0.01}

    Usage
    -----
    The user should provide a RawData object contaning no a priori calculated statistics (only 'OHLC', Volume, TickVol) along with the
    desired model-type and its parameters.  This class is intended to be used in the following flow:

    Given a .db containing ('OHLC', Volume, TickVol) time series data, one should instantiate a RawData to read the time-series and feed
    the PrimaryModel class. The class then calculates the necessary statistics the desired model-type in a curated data in the `PrimaryModel.feature_data`
    attribute. The feature_data is then used to generate the event triggers, which is used to generate labels.
    """

    def __init__(self, raw_data, model_type, parameters):

        self.model_type = model_type
        self.model_parameters = parameters["ModelParameters"]
        self.feature_data = mvp.curated.CuratedData(
            raw_data, parameters["ModelParameters"]
        )
        self.events_df = self.get_events_dataframe()

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
            columns = ['DateTime','Trigger']
            'DateTime' is the pd.index column, containing pd.Timestamp() values for the calculated events.
            'Trigger' is a coulmn containing the side of the stratgey: Buy = 1, Sell = -1.
        """
        MA_params = list(set(self.model_parameters["MA"]))
        if len(MA_params) > 2:
            print(
                "[+] Aborting strategy: Number of parameters exceeded maximum of two:"
                + print(MA_params)
            )
            return None
        delta_MA = pd.DataFrame()
        delta_MA["Delta"] = (
            self.feature_data.df_curated["MA_" + str(MA_params[0])]
            - self.feature_data.df_curated["MA_" + str(MA_params[1])]
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
            .rename(columns={"Delta": "Trigger"})
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
            columns = ['DateTime','Trigger']
            'DateTime' is the pd.index column, containing pd.Timestamp() values for the calculated events.
            'Trigger' is a coulmn containing the side of the stratgey: Buy = 1, Sell = -1.
        """
        close_data = self.feature_data.df_curated[["Close"]]
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

        return pd.DataFrame(events_CF, columns=["DateTime", "Trigger"])

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
            columns = ['DateTime','Trigger']
            'DateTime' is the pd.index column, containing pd.Timestamp() values for the calculated events.
            'Trigger' is a coulmn containing the side of the stratgey: Buy = 1, Sell = -1.
        """
        MA_param = self.model_parameters["MA"]
        if len(MA_param) > 1:
            raise IOError(
                "[+] Aborting strategy: Number of Moving Average parameters exceeded maximum of one: {}".format(
                    MA_param
                )
            )
        DEV_param = self.model_parameters["DEV"]
        if len(DEV_param) > 1:
            raise IOError(
                "[+] Aborting strategy: Number of Standard Deviation parameters exceeded maximum of one: {}".format(
                    DEV_param
                )
            )
        K_value = self.model_parameters["K_value"]
        if type(K_value) != int:
            raise IOError(
                "[+] Aborting strategy: K value parameter needs to be an integer"
            )
        temp = pd.DataFrame()
        temp["Close"] = self.feature_data.df_curated["Close"]
        temp["plusK"] = (
            self.feature_data.df_curated["MA_" + str(MA_param[0])]
            + K_value
            * self.feature_data.df_curated["DEV_" + str(DEV_param[0])]
        ).dropna()
        temp["minusK"] = (
            self.feature_data.df_curated["MA_" + str(MA_param[0])]
            - K_value
            * self.feature_data.df_curated["DEV_" + str(DEV_param[0])]
        ).dropna()

        temp["Trigger"] = temp.apply(self.states_condition, raw=True, axis=1)

        events_bol = (
            temp["Trigger"]
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
        if self.model_type == "crossing-MA":
            return self.events_crossing_MA()
        elif self.model_type == "bollinger-bands":
            return self.events_bollinger()
        elif self.model_type == "classical-filter":
            return self.events_classical_filter(
                self.model_parameters["threshold"]
            )

    def get_events_dataframe(self):
        return pd.concat(
            [
                self.feature_data.df_curated[["Close"]],
                self.events().set_index("DateTime"),
            ],
            axis=1,
        )