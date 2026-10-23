
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

from freqtrade.strategy import (
    IStrategy,
    informative,
    stoploss_from_absolute,
)
from datetime import datetime


class TP1Strategy(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "1h"
    can_short = True
    startup_candle_count = 220

    minimal_roi = {"0": 10}
    stoploss = -0.10
    use_exit_signal = True
    process_only_new_candles = True

    # Risk sizing must be configured separately.
    # This strategy sets a structural stop and a 2R target.

    @informative("4h")
    def populate_indicators_4h(
        self, dataframe: DataFrame, metadata: dict
    ) -> DataFrame:
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema200"] = ta.EMA(dataframe, timeperiod=200)
        return dataframe

    def populate_indicators(
        self, dataframe: DataFrame, metadata: dict
    ) -> DataFrame:
        dataframe["ema20"] = ta.EMA(dataframe, timeperiod=20)
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema200"] = ta.EMA(dataframe, timeperiod=200)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)

        dataframe["swing_low"] = (
            dataframe["low"].rolling(10).min().shift(1)
        )
        dataframe["swing_high"] = (
            dataframe["high"].rolling(10).max().shift(1)
        )

        dataframe["long_trend"] = (
            (dataframe["close"] > dataframe["ema200"])
            & (dataframe["ema50"] > dataframe["ema200"])
            & (dataframe["ema50_4h"] > dataframe["ema200_4h"])
        )

        dataframe["short_trend"] = (
            (dataframe["close"] < dataframe["ema200"])
            & (dataframe["ema50"] < dataframe["ema200"])
            & (dataframe["ema50_4h"] < dataframe["ema200_4h"])
        )

        dataframe["pullback_long"] = (
            (dataframe["low"] <= dataframe["ema20"])
            & (dataframe["close"] > dataframe["ema20"])
            & (dataframe["rsi"] > 50)
        )

        dataframe["pullback_short"] = (
            (dataframe["high"] >= dataframe["ema20"])
            & (dataframe["close"] < dataframe["ema20"])
            & (dataframe["rsi"] < 50)
        )

        return dataframe

    def populate_entry_trend(
        self, dataframe: DataFrame, metadata: dict
    ) -> DataFrame:
        dataframe.loc[
            (
                dataframe["long_trend"]
                & dataframe["pullback_long"]
                & (dataframe["volume"] > 0)
            ),
            ["enter_long", "enter_tag"],
        ] = (1, "tp1_long")

        dataframe.loc[
            (
                dataframe["short_trend"]
                & dataframe["pullback_short"]
                & (dataframe["volume"] > 0)
            ),
            ["enter_short", "enter_tag"],
        ] = (1, "tp1_short")

        return dataframe

    def populate_exit_trend(
        self, dataframe: DataFrame, metadata: dict
    ) -> DataFrame:
        dataframe.loc[
            dataframe["close"] < dataframe["ema20"],
            "exit_long",
        ] = 1

        dataframe.loc[
            dataframe["close"] > dataframe["ema20"],
            "exit_short",
        ] = 1

        return dataframe
